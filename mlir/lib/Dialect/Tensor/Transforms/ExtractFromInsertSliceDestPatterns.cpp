//===- ExtractFromInsertSliceDestPatterns.cpp -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/ViewLikeInterfaceUtils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::tensor;

namespace {
/// Updates extract_slice to extrace from insert_slice op's destination tensor
/// when the extract_slice and insert_slice are covering disjoint slices.
///
/// Example:
/// ```mlir
/// %i = tensor.insert_slice %src into %dst[0, 0, 0, 0][1, 1, 2, 4][1, 1, 1, 1]
///        : tensor<1x2x4xf32> into tensor<1x2x2x4xf32>
/// %e = tensor.extract_slice %i[0, 1, 0, 0] [1, 1, 2, 4] [1, 1, 1, 1]
///        : tensor<1x2x2x4xf32> to tensor<1x2x4xf32>
/// ```
/// Can be converted into
/// ```mlir
/// %i = tensor.insert_slice %src into %dst[0, 0, 0, 0][1, 1, 2, 4][1, 1, 1, 1]
///        : tensor<1x2x4xf32> into tensor<1x2x2x4xf32>
/// %e = tensor.extract_slice %dest[0, 1, 0, 0] [1, 1, 2, 4] [1, 1, 1, 1]
///        : tensor<1x2x2x4xf32> to tensor<1x2x4xf32>
/// ```
/// This helps to break the chain of insert_slice and extract_slices, which
/// might enable further optimizations.
struct ExtractFromInsertDest final : public OpRewritePattern<ExtractSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ExtractSliceOp extractOp,
                                PatternRewriter &rewriter) const override {
    auto insertOp = extractOp.getSource().getDefiningOp<InsertSliceOp>();
    if (!insertOp)
      return failure();

    if (!areDisjointSlices(insertOp, extractOp))
      return rewriter.notifyMatchFailure(extractOp, "not disjoint");

    rewriter.replaceOpWithNewOp<ExtractSliceOp>(
        extractOp, extractOp.getType(), insertOp.getDest(),
        extractOp.getMixedOffsets(), extractOp.getMixedSizes(),
        extractOp.getMixedStrides());

    return success();
  }
};
} // namespace

void mlir::tensor::populateExtractFromInsertSliceDestOpPatterns(
    RewritePatternSet &patterns) {
  patterns.add<ExtractFromInsertDest>(patterns.getContext());
}
