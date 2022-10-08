//===- HoistExtractInsertSlicePatterns.cpp --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/ViewLikeInterfaceUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "mlir-hoist-extract-insert-slice"

using namespace mlir;
using namespace mlir::tensor;

/// Verifies that the `index`-th yielded value is coming from a hoistable
/// insert_slice op and returns the insert_slice op.
static InsertSliceOp
getHoistableInsertSlice(scf::ForOp forOp, unsigned index,
                        ArrayRef<InsertSliceOp> insertOps) {
  auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  Value yieldValue = yieldOp.getOperands()[index];

  // Expect the yielded value to come from a insert_slice op.
  auto insertOp = yieldValue.getDefiningOp<InsertSliceOp>();
  if (!insertOp) {
    LLVM_DEBUG(llvm::dbgs()
               << "yielded value not coming from insert slice op\n");
    return nullptr;
  }
  if (!insertOp->hasOneUse()) {
    LLVM_DEBUG(llvm::dbgs() << "insert slice has more than one use\n");
    return nullptr;
  }
  LLVM_DEBUG(llvm::dbgs() << "yielded insert op: " << insertOp << "\n");

  // Make sure this insert_slice op is updating some loop carried value.
  // All insert_slice ops doing that is previously collected in `insertOps`.
  if (!llvm::is_contained(insertOps, insertOp)) {
    LLVM_DEBUG(llvm::dbgs()
               << "insert slice op not updating loop carried value\n");
    return nullptr;
  }

  // The destination tensor of the insert_slice op should be the block
  // argument representing the loop carried value.
  Value insertDest = insertOp.getDest();
  auto destBlockArg = insertDest.dyn_cast<BlockArgument>();
  if (!destBlockArg) {
    // Allow a chain of insert_slice ops that build upon on another. But the
    // first insert_slice op must insert into the block argument.
    while (auto prevOp = insertDest.getDefiningOp<InsertSliceOp>()) {
      insertDest = prevOp.getDest();
      destBlockArg = insertDest.dyn_cast<BlockArgument>();
    }
  }

  // Guaranteed by `insertOp` in `insertOps`. But double check:
  assert(destBlockArg && destBlockArg.getOwner()->getParentOp() == forOp);

  if (destBlockArg.getArgNumber() != index + 1) {
    LLVM_DEBUG(llvm::dbgs()
               << "index mismatch between yield and insert slice dest\n");
    return nullptr;
  }

  // All insert_slice offsets/sizes/strides must be loop invariant.
  for (Value v : insertOp->getOperands().drop_front(
           InsertSliceOp::getOffsetSizeAndStrideStartOperandIndex())) {
    if (!forOp.isDefinedOutsideOfLoop(v)) {
      LLVM_DEBUG(llvm::dbgs() << "slice offset/size/stride defined inside loop:"
                              << v << "\n");
      return nullptr;
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "hoistable insert op: " << insertOp << "\n");
  return insertOp;
}

/// Finds the extract_slice op that have the same offsets/strides/sizes as the
/// given `insertOp` from `extractOps`.
static ExtractSliceOp
findMatchingExtractSlice(InsertSliceOp insertOp,
                         ArrayRef<ExtractSliceOp> extractOps) {
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

  LLVM_DEBUG(llvm::dbgs() << "matching extract op: " << extractOp << "\n");
  return extractOp;
}

using ForOpEraseFn = std::function<void(scf::ForOp)>;

/// Hoists the `extractOp` and `insertOp` pair that updates the `index`-th loop
/// carried value out of the given `forOp` and returns the new scf.for op.
static scf::ForOp hoistExtractInsertSlice(scf::ForOp forOp, unsigned index,
                                          ExtractSliceOp extractOp,
                                          InsertSliceOp insertOp,
                                          OpBuilder &builder,
                                          const ForOpEraseFn &forOpEraseFn) {
  // Update the extract_slice op's source and move it out.
  extractOp.getSourceMutable().assign(forOp.getInitArgs()[index]);
  forOp.moveOutOfLoop(extractOp);

  // Update the terminator yielded value and move the insert_slice op out.
  auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  yieldOp->setOperand(index, insertOp.getDest());
  insertOp->moveAfter(forOp);

  // Build a new loop to additionally yield the insert_slice op's source.
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(forOp);
  NewYieldValueFn yieldFn = [&](OpBuilder &, Location,
                                ArrayRef<BlockArgument>) {
    return SmallVector<Value>{insertOp.getSource()};
  };
  auto newForOp =
      replaceLoopWithNewYields(builder, forOp, extractOp.getResult(), yieldFn);

  // Point all uses of the loop result value to the hoisted insert_slice.
  newForOp.getResult(index).replaceAllUsesWith(insertOp.getResult());
  // Fix hoisted insert_slice op's source and destination tensors.
  insertOp.getSourceMutable().assign(newForOp.getResults().back());
  insertOp.getDestMutable().assign(newForOp.getResult(index));

  forOpEraseFn(forOp);
  return newForOp;
}

/// Collects and appends all children insert_slice ops from the given `seedOp`
/// into `insertOps`, and returns true if the insert_slice op chain rooting from
/// `seeOp` does not have other users than scf.yield ops.
static bool collectInsertSliceChain(InsertSliceOp seedOp,
                                    SmallVectorImpl<InsertSliceOp> &insertOps) {
  SmallVector<InsertSliceOp> worklist;
  worklist.push_back(seedOp);
  while (!worklist.empty()) {
    InsertSliceOp insertOp = worklist.pop_back_val();
    insertOps.push_back(insertOp);
    for (Operation *user : insertOp.getResult().getUsers()) {
      if (auto userOp = dyn_cast<InsertSliceOp>(user)) {
        worklist.push_back(userOp);
      } else if (!isa<scf::YieldOp>(user)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "non extract/insert slice user of loop carried value: "
                   << *user << "\n");
        return false;
      }
    }
  }
  return true;
}

/// Hoists extract/insert slice ops that are users of the `index`-th loop
/// carried value out of the given `forOp`. Returns the new scf.for op on
/// success; returns nullptr otherwise.
static scf::ForOp hoistLoopCarriedValueUses(scf::ForOp forOp, unsigned index,
                                            OpBuilder &builder,
                                            const ForOpEraseFn &forOpEraseFn) {
  Value loopValue = forOp.getRegionIterArgs()[index];
  LLVM_DEBUG(llvm::dbgs() << "checking loop carried value #" << index << "\n");

  // Make sure the users of the loop carried value is all insert/extract
  // slice ops. This helps to simplify further logic.
  SmallVector<ExtractSliceOp> extractOps;
  SmallVector<InsertSliceOp> insertOps;
  for (Operation *user : loopValue.getUsers()) {
    if (auto op = dyn_cast<ExtractSliceOp>(user)) {
      extractOps.push_back(op);
      continue;
    }
    if (auto op = dyn_cast<InsertSliceOp>(user)) {
      if (!collectInsertSliceChain(op, insertOps))
        return nullptr;
      continue;
    }
    LLVM_DEBUG(llvm::dbgs()
               << "non extract/insert slice user of loop carried value: "
               << *user << "\n");
    return nullptr;
  }

  InsertSliceOp insertOp = getHoistableInsertSlice(forOp, index, insertOps);
  if (!insertOp)
    return nullptr;
  // To be conservative, require all other insert slice ops be disjoint with the
  // one to hoist out.
  for (InsertSliceOp otherOp : insertOps) {
    if (otherOp != insertOp && !areDisjointSlices(otherOp, insertOp)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "insert slice op not disjoint with: " << otherOp << "\n");
      return nullptr;
    }
  }

  ExtractSliceOp extractOp = findMatchingExtractSlice(insertOp, extractOps);
  if (!extractOp)
    return nullptr;
  // To be conservative, require all other extract slice ops be disjoint with
  // the one to hoist out.
  for (ExtractSliceOp otherOp : extractOps) {
    if (otherOp != extractOp && !areDisjointSlices(otherOp, extractOp)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "extract slice op not disjoint with: " << otherOp << "\n");
      return nullptr;
    }
  }

  return hoistExtractInsertSlice(forOp, index, extractOp, insertOp, builder,
                                 forOpEraseFn);
}

scf::ForOp tensor::hoistTensorExtractInsertSliceOps(scf::ForOp forOp,
                                                    OpBuilder &builder) {
  auto eraseFn = [](scf::ForOp forOp) { forOp->erase(); };
  bool changed = true;
  while (changed) {
    changed = false;
    for (unsigned i = 0; i < forOp.getNumRegionIterArgs(); ++i)
      if (auto newOp = hoistLoopCarriedValueUses(forOp, i, builder, eraseFn)) {
        forOp = newOp; // Use the new scf.for op for next iteration
        changed = true;
        break;
      }
  };
  return forOp;
}

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
    auto eraseFn = [&](scf::ForOp op) { rewriter.eraseOp(op); };
    for (unsigned i = 0; i < forOp.getNumRegionIterArgs(); ++i)
      if (hoistLoopCarriedValueUses(forOp, i, rewriter, eraseFn))
        return success();
    return failure();
  }
};

} // namespace

void tensor::populateHoistExtractInsertSliceOpPatterns(
    RewritePatternSet &patterns) {
  patterns.add<HoistExtractInsertSlice>(patterns.getContext());
}
