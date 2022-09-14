//===- HoistExtractInsertSlice.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "mlir-hoist-extract-insert-slice"

using namespace mlir;
using namespace mlir::tensor;

namespace {

/// Hoists pairs of tensor.extract_slice and tensor.insert_slice ops out of the
/// surrounding scf.for loops.
///
/// This requires the extract/insert slice op pair to have the exact same
/// loop-invariant offsets, strides, and sizes. Also they should extract from /
/// insert into the same loop carried value.
struct HoistExtractInsertSlice : OpRewritePattern<scf::ForOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    for (unsigned i = 0; i < forOp.getNumRegionIterArgs(); ++i)
      if (succeeded(hoistLoopCarriedValueUses(forOp, i, rewriter)))
        return success();

    return failure();
  }

  /// Hoists extract/insert slice ops that are users of the `index`-th loop
  /// carried value out of the given `forOp`.
  LogicalResult hoistLoopCarriedValueUses(scf::ForOp forOp, unsigned index,
                                          PatternRewriter &rewriter) const {
    Value loopValue = forOp.getRegionIterArgs()[index];
    LLVM_DEBUG(llvm::dbgs() << "inspecting loop value #" << index << "\n");
    // Make sure the users of the loop carried value is all insert/extract
    // slice ops. This helps to simplify further logic.
    SmallVector<ExtractSliceOp> extractOps;
    for (Operation *user : loopValue.getUsers()) {
      if (auto op = dyn_cast<ExtractSliceOp>(user)) {
        extractOps.push_back(op);
      } else if (!isa<InsertSliceOp>(user)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "loop carried value has non extract/insert slice user\n");
        return failure();
      }
    }

    InsertSliceOp insertOp = getHoistableInsertSlice(forOp, index);
    if (!insertOp)
      return failure();
    ExtractSliceOp extractOp = findMatchingExtractSlice(insertOp, extractOps);
    if (!extractOp)
      return failure();

    hoistExtractInsertSlice(forOp, index, extractOp, insertOp);

    return success();
  }

  /// Verifies that the `index`-th yielded value is coming from a hoistable
  /// insert_slice op and returns the insert_slice op.
  InsertSliceOp getHoistableInsertSlice(scf::ForOp forOp,
                                        unsigned index) const {
    // Expect the yielded value is coming from a insert_slice op.
    auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    Value yieldValue = yieldOp.getOperands()[index];
    auto insertOp = yieldValue.getDefiningOp<InsertSliceOp>();
    if (!insertOp) {
      LLVM_DEBUG(llvm::dbgs()
                 << "yielded value not coming from insert slice op\n");
      return nullptr;
    }
    LLVM_DEBUG(llvm::dbgs() << "last insert op: " << insertOp << "\n");

    // The destination tensor of the insert_slice op should be the block
    // argument representing the loop carried value.
    Value insertDest = insertOp.getDest();
    auto destBlockArg = insertDest.dyn_cast<BlockArgument>();
    if (!destBlockArg) {
      // Allow a chain of insert_slice ops that build upon on another. But the
      // first insert_slice op must insert into the block argument.
      while (auto prevOp = insertDest.getDefiningOp<InsertSliceOp>()) {
        LLVM_DEBUG(llvm::dbgs() << "prevous insert op: " << prevOp << "\n");
        // To be conservative, require all the previous slices they should be
        // disjoint from this one.
        if (!areDisjointSlices(prevOp, insertOp)) {
          LLVM_DEBUG(llvm::dbgs() << "insert slice op not disjoint with: "
                                  << prevOp << "\n");
          return nullptr;
        }

        insertDest = prevOp.getDest();
        destBlockArg = insertDest.dyn_cast<BlockArgument>();
      }
    }
    if (!destBlockArg) {
      LLVM_DEBUG(llvm::dbgs()
                 << "no insert slice (chain) updating loop carried value\n");
      return nullptr;
    }
    if (destBlockArg.getOwner()->getParentOp() != forOp) {
      LLVM_DEBUG(llvm::dbgs()
                 << "insert slice updating other loop's carried value\n");
      return nullptr;
    }
    if (destBlockArg.getArgNumber() != index + 1) {
      LLVM_DEBUG(llvm::dbgs()
                 << "index mismatch between yield and insert slice dest\n");
      return nullptr;
    }

    // All insert_slice offsets/sizes/strides must be loop invariant.
    for (Value v : insertOp->getOperands().drop_front(
             InsertSliceOp::getOffsetSizeAndStrideStartOperandIndex())) {
      if (!forOp.isDefinedOutsideOfLoop(v)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "slice offset/size/stride defined inside loop:" << v
                   << "\n");
        return nullptr;
      }
    }

    LLVM_DEBUG(llvm::dbgs() << "hoistable insert op: " << insertOp << "\n");
    return insertOp;
  }

  /// Finds the extract_slice op that have the same offsets/strides/sizes as the
  /// given `insertOp` from `extractOps`.
  ExtractSliceOp
  findMatchingExtractSlice(InsertSliceOp insertOp,
                           ArrayRef<ExtractSliceOp> extractOps) const {
    unsigned opIndex = 0;
    ExtractSliceOp extractOp;
    for (; opIndex < extractOps.size(); ++opIndex) {
      extractOp = extractOps[opIndex];
      const auto isSame = [](OpFoldResult a, OpFoldResult b) { return a == b; };
      if (extractOp.getType() == insertOp.getSourceType() &&
          extractOp.isSameAs(insertOp, isSame))
        break;
    }
    if (opIndex == extractOps.size()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "missing matched extract slice for yielded insert slice\n");
      return nullptr;
    }

    // To be conservative, make sure all extract_slice ops folowing this one are
    // disjoint. (We have already checked before insert_slice ops are disjoint.)
    for (++opIndex; opIndex < extractOps.size(); ++opIndex)
      if (!areDisjointSlices(extractOps[opIndex], extractOp)) {
        LLVM_DEBUG(llvm::dbgs() << "insert slice op chain not disjoint with: "
                                << extractOps[opIndex] << "\n");
        return nullptr;
      }

    LLVM_DEBUG(llvm::dbgs() << "matching extract op: " << extractOp << "\n");
    return extractOp;
  }

  /// Hoists the `extractOp` and `insertOp` pair that updates the `index`-th
  /// loop carried value out of the given `forOp`.
  void hoistExtractInsertSlice(scf::ForOp forOp, unsigned index,
                               ExtractSliceOp extractOp,
                               InsertSliceOp insertOp) const {
    // Update the extract_slice op's source and move it out.
    extractOp.getSourceMutable().assign(forOp.getInitArgs()[index]);
    forOp.moveOutOfLoop(extractOp);

    // Update the terminator yielded value and move the insert_slice op out.
    auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    yieldOp->setOperand(index, insertOp.getDest());
    insertOp->moveAfter(forOp);

    // Build a new loop to additionally yield the insert_slice op's source.
    OpBuilder builder(forOp);
    NewYieldValueFn yieldFn = [&](OpBuilder &, Location,
                                  ArrayRef<BlockArgument>) {
      return SmallVector<Value>{insertOp.getSource()};
    };
    auto newForOp = replaceLoopWithNewYields(builder, forOp,
                                             extractOp.getResult(), yieldFn);

    // Point all uses of the loop result value to the hoisted insert_slice.
    newForOp.getResult(index).replaceAllUsesWith(insertOp.getResult());
    // Fix hoisted insert_slice op's source and destination tensors.
    insertOp.getSourceMutable().assign(newForOp.getResults().back());
    insertOp.getDestMutable().assign(newForOp.getResult(index));

    forOp.erase();
  }
};

} // namespace

void tensor::populateHoistExtractInsertSliceOpPatterns(
    RewritePatternSet &patterns) {
  patterns.add<HoistExtractInsertSlice>(patterns.getContext());
}
