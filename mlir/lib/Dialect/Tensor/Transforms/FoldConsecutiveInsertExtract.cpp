//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::tensor;

/// Adds each corresponding pair of offsets in `offsets1` and `offsets2` and
/// returns the results.
static SmallVector<OpFoldResult> foldOffsets(Location loc,
                                             ArrayRef<OpFoldResult> offsets1,
                                             ArrayRef<OpFoldResult> offsets2,
                                             OpBuilder &builder) {
  SmallVector<OpFoldResult> foldedOffsets;
  assert(offsets1.size() == offsets2.size());
  foldedOffsets.reserve(offsets1.size());

  AffineExpr dim1, dim2;
  bindDims(builder.getContext(), dim1, dim2);

  for (const auto &pair : llvm::zip(offsets1, offsets2)) {
    auto offset0 =
        getValueOrCreateConstantIndexOp(builder, loc, std::get<0>(pair));
    auto offset1 =
        getValueOrCreateConstantIndexOp(builder, loc, std::get<1>(pair));
    auto foldedOffset =
        makeComposedAffineApply(builder, loc, dim1 + dim2, {offset0, offset1});
    foldedOffsets.push_back(foldedOffset.getResult());
  }
  return foldedOffsets;
}

namespace {
/// Folds consecutive tensor.extract_slice ops into one.
struct FoldConsecutiveExtractSlice : public OpRewritePattern<ExtractSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ExtractSliceOp nextOp,
                                PatternRewriter &rewriter) const override {
    auto prevOp = nextOp.getSource().getDefiningOp<ExtractSliceOp>();
    if (!prevOp)
      return failure();

    if (!prevOp.hasUnitStride() || !nextOp.hasUnitStride())
      return failure();

    auto prevResultType = prevOp.getResult().getType().cast<ShapedType>();
    if (prevOp.getSourceType().getRank() != prevResultType.getRank())
      return rewriter.notifyMatchFailure(
          prevOp, "rank-reducing producder case unimplemented");

    Location loc = nextOp.getLoc();

    SmallVector<OpFoldResult> prevOffsets = prevOp.getMixedOffsets();
    SmallVector<OpFoldResult> nextOffsets = nextOp.getMixedOffsets();
    SmallVector<OpFoldResult> foldedOffsets =
        foldOffsets(loc, prevOffsets, nextOffsets, rewriter);

    rewriter.replaceOpWithNewOp<ExtractSliceOp>(
        nextOp, nextOp.getType(), prevOp.getSource(), foldedOffsets,
        nextOp.getMixedSizes(), nextOp.getMixedStrides());
    return success();
  }
};

/// Folds consecutive tensor.insert_slice ops into one.
struct FoldConsecutiveInsertSlice : public OpRewritePattern<InsertSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(InsertSliceOp nextOp,
                                PatternRewriter &rewriter) const override {
    auto prevOp = nextOp.getSource().getDefiningOp<InsertSliceOp>();
    if (!prevOp)
      return failure();

    if (!prevOp.hasUnitStride() || !nextOp.hasUnitStride())
      return failure();

    // The first insert_slice op should be rank reducing to make sure we cover
    // the full source tensor to be inserted in the second insert_slice op.
    SliceVerificationResult result =
        isRankReducedType(prevOp.getDestType(), prevOp.getSourceType());
    if (result != SliceVerificationResult::Success)
      return failure();

    // Dynamic dimensions can pass rank reducing check in the above, e.g,
    // inserting <?xf32> into <1x?x1xf32>. For such cases we cannot be certain
    // the dynamic size covers the full tensor.
    if (!prevOp.getSourceType().hasStaticShape() ||
        !prevOp.getDestType().hasStaticShape())
      return failure();

    // All offsets in the first insert_slice op should be zeros.
    assert(prevOp.hasZeroOffset());

    rewriter.replaceOpWithNewOp<InsertSliceOp>(
        nextOp, prevOp.getSource(), nextOp.getDest(), nextOp.getMixedOffsets(),
        nextOp.getMixedSizes(), nextOp.getMixedStrides());
    return success();
  }
};
} // namespace

void mlir::tensor::populateFoldConsecutiveInsertExtractSlicePatterns(
    RewritePatternSet &patterns) {
  patterns.add<FoldConsecutiveExtractSlice, FoldConsecutiveInsertSlice>(
      patterns.getContext());
}
