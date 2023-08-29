//===- SwapExtractSliceWithFillPatterns.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::linalg;

/// Swaps tensor.extract_slice(linalg.fill(%cst, %init)) into linalg.fill(%cst,
/// tensor.extract_slice(%init)) when the linalg.fill op have no other users.
/// This helps to reduce the fill footprint.
struct SwapExtractSliceOfFill final
    : public OpRewritePattern<tensor::ExtractSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp extractOp,
                                PatternRewriter &rewriter) const override {
    auto fillOp = extractOp.getSource().getDefiningOp<FillOp>();
    if (!fillOp || !fillOp->hasOneUse())
      return failure();

    auto newExtractOp = rewriter.create<tensor::ExtractSliceOp>(
        extractOp.getLoc(), extractOp.getType(), fillOp.getOutputs()[0],
        extractOp.getMixedOffsets(), extractOp.getMixedSizes(),
        extractOp.getMixedStrides());
    rewriter.replaceOpWithNewOp<FillOp>(extractOp, fillOp.getInputs(),
                                        ValueRange{newExtractOp.getResult()});
    return success();
  }
};

/// Fold `tensor.expand/collapse_shape(linalg.fill(%val, %init))` to
/// `linalg.fill(%val, tensor.expand/collapse_shape(%init))`.
template <typename T>
struct SwapTensorShapeChangePatternsWithFillOp final
    : public OpRewritePattern<T> {
  using OpRewritePattern<T>::OpRewritePattern;

  LogicalResult matchAndRewrite(T reshapeOp,
                                PatternRewriter &rewriter) const override {
    auto fillOp = reshapeOp->getOperand(0).template getDefiningOp<FillOp>();
    if (!fillOp) {
      return failure();
    }
    auto reshapeInitOp =
        rewriter.create<T>(reshapeOp.getLoc(), reshapeOp.getResult().getType(),
                           fillOp.getDpsInitOperand(0)->get(),
                           reshapeOp.getReassociationIndices());
    rewriter.replaceOpWithNewOp<linalg::FillOp>(reshapeOp, fillOp.getInputs(),
                                                reshapeInitOp.getResult());
    return success();
  }
};

void mlir::linalg::populateFoldLinalgFillPatterns(RewritePatternSet &patterns) {
  patterns.add<SwapExtractSliceOfFill,
               SwapTensorShapeChangePatternsWithFillOp<tensor::CollapseShapeOp>,
               SwapTensorShapeChangePatternsWithFillOp<tensor::ExpandShapeOp>>(
      patterns.getContext());
}
