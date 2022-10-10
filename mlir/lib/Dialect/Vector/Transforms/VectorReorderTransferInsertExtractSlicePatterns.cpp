//===- VectorReorderTransferInsertExtractSlicePatterns.cpp ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/ViewLikeInterfaceUtils.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::vector;

namespace {

/// Reorders vector.transfer_write that are tensor.insert_slice source to be
/// after the tensor.insert_slice op.
///
/// In order to make sure the reordering is beneficial, this pattern
/// additionally requires the vector.transfer_write is writing to some
/// tensor.extract_slice that extracts from the tensor.insert_slice's
/// destination tensor.
///
/// For example, given the following IR:
/// ```
/// %extract = tensor.extract_slice %input[0, 0, 0, 0] [1, 1, 2, 4] [1, 1, 1, 1]
///              : tensor<1x2x2x4xf32> to tensor<1x2x4xf32>
/// %write0 = vector.transfer_write %val0, %extract[%c0, %c0, %c0]
///              {in_bounds = [true]} : vector<4xf32>, tensor<1x2x4xf32>
/// %write1 = vector.transfer_write %val1, %write0[%c0, %c1, %c0]
///              {in_bounds = [true]} : vector<4xf32>, tensor<1x2x4xf32>
/// %insert = tensor.insert_slice %write1 into %input[0, 0, 0, 0] [1, 1, 2, 4]
///              [1, 1, 1, 1] : tensor<1x2x4xf32> into tensor<1x2x2x4xf32>
/// ```
/// We can fold it into
/// ```mlir
/// %write0 = vector.transfer_write %val0, %input[%c0, %c0, %c0]
///              {in_bounds = [true]} : vector<4xf32>, tensor<1x2x4xf32>
/// %write1 = vector.transfer_write %val1, %write0[%c0, %c1, %c0]
///              {in_bounds = [true]} : vector<4xf32>, tensor<1x2x4xf32>
/// ```
struct ReorderTransferWriteAsInsertSource final
    : public OpRewritePattern<tensor::InsertSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::InsertSliceOp insertOp,
                                PatternRewriter &rewriter) const override {
    auto writeOp = insertOp.getSource().getDefiningOp<TransferWriteOp>();
    if (!writeOp)
      return failure();

    Value writeDest = writeOp.getSource();
    // Allow a chain of vector.transfer_write ops that build upon one another.
    // It's common to see that after vector unrolling.
    while (auto prevOp = writeDest.getDefiningOp<TransferWriteOp>())
      writeDest = prevOp.getSource();
    auto extractOp = writeDest.getDefiningOp<tensor::ExtractSliceOp>();
    if (!extractOp)
      return failure();

    // To be beneficial, require that 1) extract source to be the same as insert
    // destination; 2) the extract and insert slice op has matching offsets,
    // sizes, and strides. This makes sure they can be folded away afterwards.
    if (extractOp.getSource() != insertOp.getDest())
      return rewriter.notifyMatchFailure(insertOp, "mismatched src/dest");
    const auto isSame = [](OpFoldResult a, OpFoldResult b) { return a == b; };
    if (!extractOp.isSameAs(insertOp, isSame))
      return rewriter.notifyMatchFailure(insertOp, "mismatched parameters");

    const int64_t largeTensorRank = insertOp.getType().getRank();
    const int64_t smallTensorRank = insertOp.getSourceType().getRank();
    const int64_t vectorRank = writeOp.getVectorType().getRank();
    if (!writeOp.getPermutationMap().isMinorIdentity())
      return rewriter.notifyMatchFailure(insertOp, "not minor identity map");

    // Infer the extract/insert result tensor type (without trimming unit dims).
    RankedTensorType inferredType = extractOp.inferResultType(
        extractOp.getSourceType(), extractOp.getMixedOffsets(),
        extractOp.getMixedSizes(), extractOp.getMixedStrides());
    // Compute which dims are trimmed unit dims.
    Optional<llvm::SmallDenseSet<unsigned>> trimmedDimMask =
        computeRankReductionMask(inferredType.getShape(),
                                 extractOp.getType().getShape());
    if (!trimmedDimMask)
      return rewriter.notifyMatchFailure(insertOp, "no rank reduction mask");
    // Find the innermost trimmed dim.
    int64_t innermostTrimmedDim = -1;
    for (int64_t i = largeTensorRank - 1; i >= 0; --i)
      if (trimmedDimMask->contains(i)) {
        innermostTrimmedDim = i;
        break;
      }
    // Make sure the leading dim referenced by the vector result is not ahead of
    // the innermost trimmed dim.
    if (innermostTrimmedDim + 1 + vectorRank > largeTensorRank)
      return rewriter.notifyMatchFailure(insertOp, "unsupported rank-reducing");

    Location loc = insertOp.getLoc();
    auto newInsertOp = rewriter.create<tensor::InsertSliceOp>(
        loc, writeOp.getSource(), insertOp.getDest(),
        insertOp.getMixedOffsets(), insertOp.getMixedSizes(),
        insertOp.getMixedStrides());

    AffineExpr dim0, dim1;
    bindDims(getContext(), dim0, dim1);

    SmallVector<Value> newIndices;
    newIndices.reserve(largeTensorRank);

    int64_t transferDimIndex = 0;
    for (int64_t i = 0; i < largeTensorRank; ++i) {
      Value offset = getValueOrCreateConstantIndexOp(
          rewriter, loc, insertOp.getMixedOffsets()[i]);
      if (trimmedDimMask->contains(i)) {
        // For unit dims trimmed by the extract/insert slice op, the index would
        // directly be its original extract/insert offset.
        newIndices.push_back(offset);
      } else {
        // For other dims, we need to add the transfer_write's offset.
        newIndices.push_back(makeComposedAffineApply(
            rewriter, loc, dim0 + dim1,
            {writeOp.getIndices()[transferDimIndex++], offset}));
      }
    }
    assert(transferDimIndex == smallTensorRank);

    auto newMap = AffineMap::getMinorIdentityMap(largeTensorRank, vectorRank,
                                                 writeOp.getContext());
    rewriter.replaceOpWithNewOp<TransferWriteOp>(
        insertOp, writeOp.getVector(), newInsertOp.getResult(), newIndices,
        AffineMapAttr::get(newMap), writeOp.getMask(),
        writeOp.getInBoundsAttr());
    return success();
  }
};

/// Reorders vector.transfer_write that are tensor.insert_slice destination to
/// be after the tensor.insert_slice op when the ranges are disjoint.
///
/// E.g., the following IR:
/// ``mlir
/// %0 = vector.transfer_write %val, %src[0, 0, 1, 0] {in_bounds = [true]}
///        : vector<4xf32>, tensor<1x2x2x4xf32>
/// %1 = tensor.insert_slice %slice into %0[0, 1, 0, 0] [1, 1, 2, 4]
///        [1, 1, 1, 1] : tensor<1x2x4xf32> into tensor<1x2x2x4xf32>
/// ```
/// Can be converted into
/// ```mlir
/// %0 = tensor.insert_slice %slice into %src[0, 1, 0, 0] [1, 1, 2, 4]
///        [1, 1, 1, 1] : tensor<1x2x4xf32> into tensor<1x2x2x4xf32>
/// %1 = vector.transfer_write %val, %0[0, 0, 1, 0] {in_bounds = [true]}
///        : vector<4xf32>, tensor<1x2x2x4xf32>
/// ```
struct ReorderTransferWriteAsInsertDest final
    : public OpRewritePattern<tensor::InsertSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::InsertSliceOp insertOp,
                                PatternRewriter &rewriter) const override {
    auto writeOp = insertOp.getDest().getDefiningOp<TransferWriteOp>();
    if (!writeOp)
      return rewriter.notifyMatchFailure(insertOp, "not inserting into write");
    if (!writeOp.getPermutationMap().isMinorIdentity())
      return rewriter.notifyMatchFailure(insertOp, "not minor identity map");

    if (!insertOp.hasUnitStride())
      return rewriter.notifyMatchFailure(insertOp, "not unit stride");

    const unsigned writeTensorRank = writeOp.getSource().getType().getRank();
    const unsigned writeReducedRank = writeOp.getLeadingShapedRank();

    SmallVector<OpFoldResult> writeOffsets;
    writeOffsets.reserve(writeTensorRank);
    llvm::append_range(writeOffsets, writeOp.getIndices());

    SmallVector<OpFoldResult> writeSizes;
    writeSizes.reserve(writeTensorRank);
    for (unsigned i = 0; i < writeReducedRank; ++i)
      writeSizes.push_back(rewriter.getIndexAttr(1));
    for (unsigned i = writeReducedRank; i < writeTensorRank; ++i)
      writeSizes.push_back(rewriter.getIndexAttr(
          writeOp.getVectorType().getDimSize(i - writeReducedRank)));

    SmallVector<OpFoldResult> insertOffsets = insertOp.getMixedOffsets();
    SmallVector<OpFoldResult> insertSizes = insertOp.getMixedSizes();

    if (!areDisjointRanges(writeOffsets, writeSizes, insertOffsets,
                           insertSizes))
      return rewriter.notifyMatchFailure(insertOp, "not disjoint ranges");

    auto newInsertOp = rewriter.create<tensor::InsertSliceOp>(
        insertOp.getLoc(), insertOp.getSource(), writeOp.getSource(),
        insertOp.getMixedOffsets(), insertOp.getMixedSizes(),
        insertOp.getMixedStrides());

    rewriter.replaceOpWithNewOp<TransferWriteOp>(
        insertOp, writeOp.getVector(), newInsertOp.getResult(),
        writeOp.getIndices(), writeOp.getPermutationMapAttr(),
        writeOp.getMask(), writeOp.getInBoundsAttr());

    return success();
  }
};

} // namespace

void vector::populateVectorReorderTransferExtractInsertSlicePatterns(
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<ReorderTransferWriteAsInsertSource,
               ReorderTransferWriteAsInsertDest>(patterns.getContext(),
                                                 benefit);
}
