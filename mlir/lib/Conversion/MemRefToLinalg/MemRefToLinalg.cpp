//===- MemRefToLinalg.cpp - MemRef to Linalg dialect conversion -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/MemRefToLinalg/MemRefToLinalg.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTMEMREFTOLINALG
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

/// Create a linalg::GenericOp version of an n-D copy that can further tile,
/// lower to loops or vectorize, unlike the current implementation of
/// memref::CopyOp.
static Operation *createLinalgCopyOp(OpBuilder &b, Location loc, Value from,
                                     Value to,
                                     ArrayRef<NamedAttribute> attributes) {
  auto memrefTypeFrom = from.getType().dyn_cast<MemRefType>();
  auto memrefTypeTo = to.getType().dyn_cast<MemRefType>();
  if (!memrefTypeFrom || !memrefTypeTo ||
      memrefTypeFrom.getRank() != memrefTypeTo.getRank()) {
    mlir::emitError(
        loc, "unable to generate copy op within bufferization from type ")
        << memrefTypeFrom << " to " << memrefTypeTo;
    return nullptr;
  }
  AffineMap id =
      AffineMap::getMultiDimIdentityMap(memrefTypeTo.getRank(), b.getContext());
  SmallVector<utils::IteratorType> iteratorTypes(memrefTypeTo.getRank(),
                                                 utils::IteratorType::parallel);
  return b.create<linalg::GenericOp>(
      loc,
      /*inputs=*/from,
      /*outputs=*/to,
      /*indexingMaps=*/llvm::ArrayRef<AffineMap>({id, id}),
      /*iteratorTypes=*/iteratorTypes,
      [](OpBuilder &b, Location loc, ValueRange args) {
        b.create<linalg::YieldOp>(loc, args.front());
      },
      attributes);
}

namespace {
struct MemrefCopyOpToLinalg : public OpRewritePattern<memref::CopyOp> {
  using OpRewritePattern<memref::CopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CopyOp copyOp,
                                PatternRewriter &rewriter) const override {
    Operation *linalgCopy =
        createLinalgCopyOp(rewriter, copyOp.getLoc(), copyOp.getSource(),
                           copyOp.getTarget(), copyOp->getAttrs());
    if (!linalgCopy)
      return failure();
    rewriter.replaceOp(copyOp, linalgCopy->getResults());
    return success();
  }
};

struct MemRefCopyToLinalgPass
    : public impl::ConvertMemrefToLinalgBase<MemRefCopyToLinalgPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(&getContext());
    patterns.insert<MemrefCopyOpToLinalg>(context);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createConvertMemRefToLinalgPass() {
  return std::make_unique<MemRefCopyToLinalgPass>();
}
