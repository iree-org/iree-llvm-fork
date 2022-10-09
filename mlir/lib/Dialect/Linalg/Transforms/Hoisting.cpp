//===- Hoisting.cpp - Linalg hoisting transformations ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements functions concerned with hoisting invariant operations
// in the context of Linalg transformations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"

using llvm::dbgs;

#define DEBUG_TYPE "linalg-hoisting"

#define DBGS() (dbgs() << '[' << DEBUG_TYPE << "] ")

using namespace mlir;
using namespace mlir::linalg;

/// Look for a vector.transfer_read, in the uses of the given `srcTensor`,
/// accessing the same offset as the vector.transfer_write.
static vector::TransferReadOp
findMatchingTransferRead(vector::TransferWriteOp write, Value srcTensor) {
  LLVM_DEBUG(DBGS() << "findMatchingTransferRead for: " << write << "\n");
  SmallVector<Operation *> users = llvm::to_vector(srcTensor.getUsers());
  while (!users.empty()) {
    Operation *user = users.pop_back_val();
    LLVM_DEBUG(DBGS() << "inspect potential read user: " << *user << "\n");

    auto read = dyn_cast<vector::TransferReadOp>(user);
    if (read && read.getIndices() == write.getIndices() &&
        read.getVectorType() == write.getVectorType())
      return read;

    if (isa<vector::TransferWriteOp>(user)) {
      // If we find a write with disjoint indices recurse through its uses.
      if (vector::isDisjointTransferIndices(
              cast<VectorTransferOpInterface>(user),
              cast<VectorTransferOpInterface>(*write))) {
        users.append(user->getUsers().begin(), user->getUsers().end());
      }
    }
  }
  return nullptr;
}

/// Return true if the chunk of data inserted by the vector.transfer_write op
/// are read by any other op than the vector.transfer_read candidate.
static bool tensorChunkAccessedByUnknownOp(vector::TransferWriteOp write,
                                           vector::TransferReadOp candidateRead,
                                           BlockArgument tensorArg) {
  // Make sure none of the other uses read the part of the tensor modified
  // by the transfer_write.
  llvm::SmallVector<Value::use_range, 1> uses;
  uses.push_back(tensorArg.getUses());
  while (!uses.empty()) {
    for (OpOperand &use : uses.pop_back_val()) {
      Operation *user = use.getOwner();
      // Skip the candidate use, only inspect the "other" uses.
      if (user == candidateRead || user == write)
        continue;
      // Tensor extract/insert slice ops should be hoisted separately. Just bail
      // out if we see them here.
      if (isa<tensor::ExtractSliceOp, tensor::InsertSliceOp>(user))
        return true;
      // Consider all transitive uses through a vector.transfer_write.
      if (auto writeUser = dyn_cast<vector::TransferWriteOp>(user)) {
        uses.push_back(writeUser->getResult(0).getUses());
        continue;
      }
      // Consider all nested uses through an scf::ForOp. We may have
      // pass-through tensor arguments left from previous level of
      // hoisting.
      if (auto forUser = dyn_cast<scf::ForOp>(user)) {
        Value arg = forUser.getLoopBody().getArgument(
            use.getOperandNumber() - forUser.getNumControlOperands() +
            /*iv value*/ 1);
        uses.push_back(arg.getUses());
        continue;
      }
      // Follow the use yield as long as it doesn't escape the original
      // region.
      if (auto yieldUser = dyn_cast<scf::YieldOp>(user)) {
        Operation *yieldParent = yieldUser->getParentOp();
        if (write->getParentOp()->isAncestor(yieldParent)) {
          Value ret = yieldParent->getResult(use.getOperandNumber());
          uses.push_back(ret.getUses());
          continue;
        }
      }
      auto read = dyn_cast<vector::TransferReadOp>(user);
      if (!read || !vector::isDisjointTransferIndices(
                       cast<VectorTransferOpInterface>(*read),
                       cast<VectorTransferOpInterface>(*write))) {
        return true;
      }
    }
  }
  return false;
}

/// Return the `forOp`-invariant vector.transfer_write that produces the given
/// `yieldOperand`. Return nullptr if `yieldOperand` is not produced by a
/// vector.transfer_write op, or if any of the indexings `forOp`-dependent.
static vector::TransferWriteOp
getLoopInvariantTransferWrite(scf::ForOp forOp, OpOperand &yieldOperand) {
  Value v = yieldOperand.get();
  if (auto write = v.getDefiningOp<vector::TransferWriteOp>()) {
    // Indexing must not depend on `forOp`.
    for (Value operand : write.getIndices())
      if (!forOp.isDefinedOutsideOfLoop(operand))
        return nullptr;
    return write;
  }

  return nullptr;
}

/// Mechanically hoist matching vector transfer read/write pairs involving
/// `tensorBBArg` out of the enclosing parent scf.for op.
static void hoistReadWrite(vector::TransferReadOp read,
                           vector::TransferWriteOp write,
                           BlockArgument tensorBBArg) {
  scf::ForOp forOp = cast<scf::ForOp>(tensorBBArg.getOwner()->getParentOp());
  assert(read && write && "expected valid transfer_read and transfer_write");
  LLVM_DEBUG(DBGS() << "In forOp:\n"
                    << *forOp.getOperation() << "\nHoist: " << read
                    << "\nHoist: " << write << "\nInvolving: " << tensorBBArg
                    << "\n");

  // Hoist the transfer_read op.
  forOp.moveOutOfLoop(read);

  // TODO: don't hardcode /*numIvs=*/1.
  assert(tensorBBArg.getArgNumber() >= /*numIvs=*/1);
  unsigned initArgNumber = tensorBBArg.getArgNumber() - /*numIvs=*/1;

  // Update the source tensor.
  read.getSourceMutable().assign(forOp.getInitArgs()[initArgNumber]);

  // Hoist write after.
  write->moveAfter(forOp);

  // Update the yield.
  auto yieldOp = cast<scf::YieldOp>(forOp.getRegion().front().getTerminator());
  yieldOp->setOperand(initArgNumber, write.getSource());

  // Rewrite `loop` with additional new yields.
  OpBuilder b(read);
  NewYieldValueFn yieldFn = [&](OpBuilder &b, Location loc,
                                ArrayRef<BlockArgument> newBBArgs) {
    return SmallVector<Value>{write.getVector()};
  };
  auto newForOp = replaceLoopWithNewYields(b, forOp, read.getVector(), yieldFn);

  // Transfer write has been hoisted, need to update the vector and tensor
  // source. Replace the result of the loop to use the new tensor created
  // outside the loop.
  // Depending on whether a insert_slice is present or not, it carries the
  // update on the tensor operands.
  newForOp.getResult(initArgNumber).replaceAllUsesWith(write.getResult());
  write.getSourceMutable().assign(newForOp.getResult(initArgNumber));

  // Always update with the newly yield tensor and vector.
  write.getVectorMutable().assign(newForOp.getResults().back());
}

// To hoist transfer op on tensor the logic can be significantly simplified
// compared to the case on buffer. The transformation follows this logic:
// 1. Look for transfer_write with a single use from ForOp yield
// 2. Check the uses of the matching block argument and look for a transfer_read
// with the same indices.
// 3. Check that all the other uses of the tensor argument are either disjoint
// tensor_read or transfer_write. For transfer_write uses recurse to make sure
// the new tensor has the same restrictions on its uses.
// 4. Hoist the tensor_read/tensor_write and update the tensor SSA links.
// After this transformation the scf.forOp may have unused arguments that can be
// remove by the canonicalization pass.
void mlir::linalg::hoistRedundantVectorTransfersOnTensor(func::FuncOp func) {
  bool changed = true;
  while (changed) {
    changed = false;
    func.walk([&](scf::ForOp forOp) {
      // Hoist tensor extract/insert slices out first.
      LLVM_DEBUG(llvm::dbgs()
                 << "before hoisting tensor slice: " << forOp << "\n");
      OpBuilder builder(forOp);
      auto newForOp = tensor::hoistTensorExtractInsertSliceOps(forOp, builder);
      if ((changed = (newForOp != forOp)))
        forOp = newForOp;
      LLVM_DEBUG(llvm::dbgs()
                 << "after hoisting tensor slice: " << forOp << "\n");

      Operation *yield = forOp.getBody()->getTerminator();
      for (const auto &it : llvm::enumerate(forOp.getRegionIterArgs())) {
        OpOperand &ret = yield->getOpOperand(it.index());
        vector::TransferWriteOp write =
            getLoopInvariantTransferWrite(forOp, ret);
        if (!write || !write->hasOneUse())
          continue;
        LLVM_DEBUG(dbgs() << "\n";
                   DBGS() << "Candidate write for hoisting: " << write << "\n");
        if (llvm::any_of(write.getIndices(), [&forOp](Value index) {
              return !forOp.isDefinedOutsideOfLoop(index);
            }))
          continue;
        // Find a read with the same type and indices.
        vector::TransferReadOp matchingRead =
            findMatchingTransferRead(write, it.value());
        // Make sure none of the other uses read the part of the tensor modified
        // by the transfer_write.
        if (!matchingRead ||
            tensorChunkAccessedByUnknownOp(write, matchingRead, it.value()))
          continue;

        LLVM_DEBUG(DBGS() << "Start hoisting\n");
        hoistReadWrite(matchingRead, write, it.value());
        changed = true;
        forOp.erase();

        // Need to interrupt and restart: erasing the loop messes up the walk.
        return WalkResult::interrupt();
      }

      return changed ? WalkResult::interrupt() : WalkResult::advance();
    });
    // Apply canonicalization so the newForOp + yield folds immediately, thus
    // cleaning up the IR and potentially enabling more hoisting.
    if (changed) {
      RewritePatternSet patterns(func->getContext());
      scf::ForOp::getCanonicalizationPatterns(patterns, func->getContext());
      (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
    }
  }
}

void mlir::linalg::hoistRedundantVectorTransfers(func::FuncOp func) {
  bool changed = true;
  while (changed) {
    changed = false;
    // First move loop invariant ops outside of their loop. This needs to be
    // done before as we cannot move ops without interrupting the function walk.
    func.walk(
        [&](LoopLikeOpInterface loopLike) { moveLoopInvariantCode(loopLike); });

    func.walk([&](vector::TransferReadOp transferRead) {
      if (!transferRead.getShapedType().isa<MemRefType>())
        return WalkResult::advance();

      LLVM_DEBUG(DBGS() << "Candidate for hoisting: "
                        << *transferRead.getOperation() << "\n");
      auto loop = dyn_cast<scf::ForOp>(transferRead->getParentOp());
      LLVM_DEBUG(DBGS() << "Parent op: " << *transferRead->getParentOp()
                        << "\n");
      if (!loop)
        return WalkResult::advance();

      LLVM_DEBUG(DBGS() << "Candidate read: " << *transferRead.getOperation()
                        << "\n");

      SetVector<Operation *> forwardSlice;
      getForwardSlice(transferRead.getOperation(), &forwardSlice);

      // Look for the last TransferWriteOp in the forwardSlice of
      // `transferRead` that operates on the same memref.
      vector::TransferWriteOp transferWrite;
      for (auto *sliceOp : llvm::reverse(forwardSlice)) {
        auto candidateWrite = dyn_cast<vector::TransferWriteOp>(sliceOp);
        if (!candidateWrite ||
            candidateWrite.getSource() != transferRead.getSource())
          continue;
        transferWrite = candidateWrite;
      }

      // All operands of the TransferRead must be defined outside of the loop.
      for (auto operand : transferRead.getOperands())
        if (!loop.isDefinedOutsideOfLoop(operand))
          return WalkResult::advance();

      // Only hoist transfer_read / transfer_write pairs for now.
      if (!transferWrite)
        return WalkResult::advance();

      LLVM_DEBUG(DBGS() << "Candidate: " << *transferWrite.getOperation()
                        << "\n");

      // Approximate aliasing by checking that:
      //   1. indices are the same,
      //   2. no other operations in the loop access the same memref except
      //      for transfer_read/transfer_write accessing statically disjoint
      //      slices.
      if (transferRead.getIndices() != transferWrite.getIndices() &&
          transferRead.getVectorType() == transferWrite.getVectorType())
        return WalkResult::advance();

      // TODO: may want to memoize this information for performance but it
      // likely gets invalidated often.
      DominanceInfo dom(loop);
      if (!dom.properlyDominates(transferRead.getOperation(), transferWrite))
        return WalkResult::advance();
      for (auto &use : transferRead.getSource().getUses()) {
        if (!loop->isAncestor(use.getOwner()))
          continue;
        if (use.getOwner() == transferRead.getOperation() ||
            use.getOwner() == transferWrite.getOperation())
          continue;
        if (auto transferWriteUse =
                dyn_cast<vector::TransferWriteOp>(use.getOwner())) {
          if (!vector::isDisjointTransferSet(
                  cast<VectorTransferOpInterface>(transferWrite.getOperation()),
                  cast<VectorTransferOpInterface>(
                      transferWriteUse.getOperation())))
            return WalkResult::advance();
        } else if (auto transferReadUse =
                       dyn_cast<vector::TransferReadOp>(use.getOwner())) {
          if (!vector::isDisjointTransferSet(
                  cast<VectorTransferOpInterface>(transferWrite.getOperation()),
                  cast<VectorTransferOpInterface>(
                      transferReadUse.getOperation())))
            return WalkResult::advance();
        } else {
          // Unknown use, we cannot prove that it doesn't alias with the
          // transferRead/transferWrite operations.
          return WalkResult::advance();
        }
      }

      // Hoist read before.
      loop.moveOutOfLoop(transferRead);

      // Hoist write after.
      transferWrite->moveAfter(loop);

      // Rewrite `loop` with new yields by cloning and erase the original loop.
      OpBuilder b(transferRead);
      NewYieldValueFn yieldFn = [&](OpBuilder &b, Location loc,
                                    ArrayRef<BlockArgument> newBBArgs) {
        return SmallVector<Value>{transferWrite.getVector()};
      };
      auto newForOp =
          replaceLoopWithNewYields(b, loop, transferRead.getVector(), yieldFn);

      // Transfer write has been hoisted, need to update the written vector by
      // the value yielded by the newForOp.
      transferWrite.getVectorMutable().assign(newForOp.getResults().back());

      changed = true;
      loop.erase();
      // Need to interrupt and restart because erasing the loop messes up the
      // walk.
      return WalkResult::interrupt();
    });
  }
}
