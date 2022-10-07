//===- VectorDropTrailingUnitDim.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

static VectorType trimTrailingOneDims(VectorType oldType) {
  ArrayRef<int64_t> oldShape = oldType.getShape();
  ArrayRef<int64_t> newShape = oldShape;
  while (!newShape.empty() && newShape.back() == 1)
    newShape = newShape.drop_back();
  // Make sure we have at least 1 dimension per vector type requirements.
  if (newShape.empty())
    newShape = oldShape.take_back();
  return VectorType::get(newShape, oldType.getElementType());
}

namespace {
struct CastAwayFmaTrailingUnitDim : public OpRewritePattern<vector::FMAOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::FMAOp fmaOp,
                                PatternRewriter &rewriter) const override {
    VectorType oldType = fmaOp.getVectorType();
    VectorType newType = trimTrailingOneDims(oldType);
    if (oldType.getRank() == newType.getRank())
      return failure();

    Location loc = fmaOp.getLoc();
    SmallVector<Value> newOperands;
    newOperands.reserve(3);
    for (Value oldOperand : fmaOp->getOperands()) {
      newOperands.push_back(
          rewriter.create<vector::ShapeCastOp>(loc, newType, oldOperand));
    }
    Value newOp = rewriter.create<vector::FMAOp>(loc, newOperands);
    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(fmaOp, oldType, newOp);
    return success();
  }
};

struct CastAwayInsertTrailingUnitDim
    : public OpRewritePattern<vector::InsertOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::InsertOp insertOp,
                                PatternRewriter &rewriter) const override {
    VectorType oldType = insertOp.getDestVectorType();
    VectorType newType = trimTrailingOneDims(oldType);
    if (oldType.getRank() == newType.getRank())
      return failure();

    int64_t numTrimmedDims = oldType.getRank() - newType.getRank();
    Location loc = insertOp.getLoc();

    auto srcVectorType = insertOp.getSourceType().dyn_cast<VectorType>();
    if (!srcVectorType) {
      // Inserting a scalar value into a vector. Just cast away trailing one
      // dims in the destination and result vector.
      Value newDest = rewriter.create<vector::ShapeCastOp>(loc, newType,
                                                           insertOp.getDest());
      ArrayRef<Attribute> positions =
          insertOp.getPosition().getValue().drop_back(numTrimmedDims);
      Value newInsert = rewriter.create<vector::InsertOp>(
          loc, insertOp.getSource(), newDest, rewriter.getArrayAttr(positions));
      rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(
          insertOp, insertOp.getDestVectorType(), newInsert);
      return success();
    }

    if (srcVectorType.getNumElements() == 1) {
      // Insert a single value vector into a vector. Extract the single value
      // and rely on the above case to cast away the trailing one dims.
      SmallVector<int64_t> extractPos(srcVectorType.getRank(), 0);
      Value newSrc = rewriter.create<vector::ExtractOp>(
          loc, srcVectorType.getElementType(), insertOp.getSource(),
          rewriter.getI64ArrayAttr(extractPos));

      SmallVector<Attribute> insertPos =
          llvm::to_vector(insertOp.getPosition().getValue());
      int64_t rank = insertOp.getDestVectorType().getRank();
      if (insertPos.size() < rank) {
        auto zero = rewriter.getI64IntegerAttr(0);
        insertPos.append(rank - insertPos.size(), zero);
      }
      rewriter.replaceOpWithNewOp<vector::InsertOp>(
          insertOp, newSrc, insertOp.getDest(),
          rewriter.getArrayAttr(insertPos));
      return success();
    }

    return failure();
  }
};

struct CastAwayExtractTrailingUnitDim
    : public OpRewritePattern<vector::ExtractOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ExtractOp extractOp,
                                PatternRewriter &rewriter) const override {
    VectorType oldType = extractOp.getType().dyn_cast<VectorType>();
    if (!oldType)
      return failure();

    VectorType newType = trimTrailingOneDims(oldType);
    if (oldType.getRank() == newType.getRank())
      return failure();

    int64_t numTrimmedDims = oldType.getRank() - newType.getRank();
    Location loc = extractOp.getLoc();
    auto newSrcType = VectorType::get(
        extractOp.getVectorType().getShape().drop_back(numTrimmedDims),
        extractOp.getVectorType().getElementType());
    Value newSrc = rewriter.create<vector::ShapeCastOp>(loc, newSrcType,
                                                        extractOp.getVector());
    Value newExtract = rewriter.create<vector::ExtractOp>(
        loc, newType, newSrc, extractOp.getPosition());
    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(
        extractOp, extractOp.getType(), newSrc);
    return success();
  }
};

struct CastAwayInsertStridedSliceTrailingUnitDim
    : public OpRewritePattern<vector::InsertStridedSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::InsertStridedSliceOp insertOp,
                                PatternRewriter &rewriter) const override {
    return success();
  }
};

struct SwapShapeCastOfBroadcast : public OpRewritePattern<vector::ShapeCastOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ShapeCastOp sCastOp,
                                PatternRewriter &rewriter) const override {
    auto bCastOp = sCastOp.getSource().getDefiningOp<vector::BroadcastOp>();
    if (!bCastOp)
      return failure();
    // Shape cast ops cannot take scalar values as input.
    auto bCastSrcVectorType = bCastOp.getSourceType().dyn_cast<VectorType>();
    if (!bCastSrcVectorType)
      return failure();

    // Restrict to trailing unit dim removal shape cast ops.
    if (trimTrailingOneDims(sCastOp.getSourceVectorType()) !=
        sCastOp.getResultVectorType())
      return failure();

    int64_t numTrimmedDims = sCastOp.getSourceVectorType().getRank() -
                             sCastOp.getResultVectorType().getRank();
    auto newSCastResultType =
        VectorType::get(bCastSrcVectorType.getShape().drop_back(numTrimmedDims),
                        bCastSrcVectorType.getElementType());

    Value newSCastOp = rewriter.create<vector::ShapeCastOp>(
        sCastOp.getLoc(), newSCastResultType, bCastOp.getSource());
    rewriter.replaceOpWithNewOp<vector::BroadcastOp>(
        sCastOp, sCastOp.getResultVectorType(), newSCastOp);
    return success();
  }
};

} // namespace

void vector::populateCastAwayVectorTrailingOneDimPatterns(
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<CastAwayFmaTrailingUnitDim, CastAwayInsertTrailingUnitDim,
               CastAwayExtractTrailingUnitDim, SwapShapeCastOfBroadcast>(
      patterns.getContext(), benefit);
}
