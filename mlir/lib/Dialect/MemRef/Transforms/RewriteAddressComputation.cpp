//===- RewriteAddressComputation.cpp - Rewrite address computation -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transformation pass rewrites load/store operations into subviews +
// load/store such that the offsets of resulting load/store are zeros.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "rewrite-address-computation"

namespace mlir {
namespace memref {
#define GEN_PASS_DEF_REWRITEADDRESSCOMPUTATION
#include "mlir/Dialect/MemRef/Transforms/Passes.h.inc"
} // namespace memref
} // namespace mlir
using namespace mlir;

//===----------------------------------------------------------------------===//
// Pass registration
//===----------------------------------------------------------------------===//

namespace {

struct RewriteAddressComputationPass final
    : public memref::impl::RewriteAddressComputationBase<
          RewriteAddressComputationPass> {
  void runOnOperation() override;
};

} // namespace

// Rewrite a load so that all its indices are zeros.
// E.g., %ld = memref.load %base[%off0]...[%offN]
// =>
// %new_base = subview %base[%off0,.., %offN][1,..,1][1,..,1]
// %ld = memref.load %new_base[0,..,0] :
//    memref<1x..x1xTy, strided<[1,..,1], offset: ?>>
//
// Ultimately we want to produce an affine map with the address computation.
// This will be taken care of by the expand-strided-metadata pass.
static void rewriteLoad(RewriterBase &rewriter, memref::LoadOp loadOp) {
  MemRefType ldTy = loadOp.getMemRefType();
  unsigned loadRank = ldTy.getRank();
  // Don't waste compile time if there is nothing to rewrite.
  if (loadRank == 0)
    return;

  RewriterBase::InsertionGuard guard(RewriteAddressComputationPass);
  rewriter.setInsertionPoint(loadOp);
  // Create the array of ones of the right size.
  SmallVector<OpFoldResult> ones(loadRank, rewriter.getIndexAttr(1));
  Location loc = loadOp.getLoc();
  auto subview = rewriter.create<memref::SubViewOp>(
      loc, /*source=*/loadOp.getMemRef(),
      /*offsets=*/getAsOpFoldResult(loadOp.getIndices()),
      /*sizes=*/ones, /*strides=*/ones);
  // Rewrite the load with the subview as the base pointer.
  SmallVector<Value> zeros(loadRank,
                           rewriter.create<arith::ConstantIndexOp>(loc, 0));
  auto newLoad = rewriter.create<memref::LoadOp>(loc, subview.getResult(),
                                                 /*indices=*/zeros);
  rewriter.replaceOp(loadOp, newLoad.getResult());
}

void RewriteAddressComputationPass::runOnOperation() {
  Operation *funcOp = getOperation();
  IRRewriter rewriter(&getContext());
  funcOp->walk([&](memref::LoadOp loadOp) {
    LLVM_DEBUG(llvm::dbgs() << "Found load:\n" << loadOp << '\n');
    rewriteLoad(rewriter, loadOp);
  });
}

std::unique_ptr<Pass> memref::createRewriteAddressComputationPass() {
  return std::make_unique<RewriteAddressComputationPass>();
}
