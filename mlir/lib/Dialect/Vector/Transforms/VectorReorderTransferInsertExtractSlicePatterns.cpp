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

/// Returns true if all rank reduced in the given `extractOp` happen in leading
/// dimensions earlier than last `trailingRank` dimensions.
static bool areAllRankReducedLeadingDim(tensor::ExtractSliceOp extractOp,
                                        unsigned trailingRank) {
  if (extractOp.getSourceType().getRank() == extractOp.getType().getRank())
    return true;

  RankedTensorType inferredType = extractOp.inferResultType(
      extractOp.getSourceType(), extractOp.getMixedOffsets(),
      extractOp.getMixedSizes(), extractOp.getMixedStrides());
  return extractOp.getType().getShape().take_back(trailingRank) ==
         inferredType.getShape().take_back(trailingRank);
}

/// Returns true if all rank reduced in the given `insertOp` happen in leading
/// dimensions earlier than last `trailingRank` dimensions.
static bool areAllRankReducedLeadingDim(tensor::InsertSliceOp insertOp,
                                        unsigned trailingRank) {
  // If no reduced ranks then simply return true.
  if (insertOp.getSourceType().getRank() == insertOp.getDestType().getRank())
    return true;

  // Infer the small type by extracting from the large type.
  RankedTensorType inferredType = tensor::ExtractSliceOp::inferResultType(
      insertOp.getDestType(), insertOp.getMixedOffsets(),
      insertOp.getMixedSizes(), insertOp.getMixedStrides());
  return insertOp.getSourceType().getShape().take_back(trailingRank) ==
         inferredType.getShape().take_back(trailingRank);
}

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
      return failure();
    const auto isSame = [](OpFoldResult a, OpFoldResult b) { return a == b; };
    if (!extractOp.isSameAs(insertOp, isSame))
      return rewriter.notifyMatchFailure(insertOp, "mismatched parameters");

    // Make sure the transfer_write op has minor identity and all reduced rank
    // are in leading dimensions. This avoid complicated rank reducing issues
    // when swap the transfer and slice op.
    int64_t largeTensorRank = insertOp.getType().getRank();
    int64_t smallTensorRank = insertOp.getSourceType().getRank();
    int64_t vectorRank = writeOp.getVectorType().getRank();
    if (!writeOp.getPermutationMap().isMinorIdentity())
      return rewriter.notifyMatchFailure(insertOp, "not minor identity map");
    if (!areAllRankReducedLeadingDim(extractOp, smallTensorRank))
      return rewriter.notifyMatchFailure(insertOp, "not leading rank reduced");

    Location loc = insertOp.getLoc();
    auto newInsertOp = rewriter.create<tensor::InsertSliceOp>(
        loc, writeOp.getSource(), insertOp.getDest(),
        insertOp.getMixedOffsets(), insertOp.getMixedSizes(),
        insertOp.getMixedStrides());

    // Prepend zeros to the indices to match the large tensor, if the extract
    // slice op is rank reducing.
    SmallVector<Value> newIndices;
    newIndices.reserve(largeTensorRank);
    int64_t reducedRank = largeTensorRank - smallTensorRank;
    for (int i = 0; i < reducedRank; ++i) {
      OpFoldResult offset = insertOp.getMixedOffsets()[i];
      newIndices.push_back(
          getValueOrCreateConstantIndexOp(rewriter, loc, offset));
    }
    AffineExpr dim0, dim1;
    bindDims(getContext(), dim0, dim1);
    for (int i = 0; i < smallTensorRank; ++i) {
      OpFoldResult offset = insertOp.getMixedOffsets()[i + reducedRank];
      Value offsetVal = getValueOrCreateConstantIndexOp(rewriter, loc, offset);
      newIndices.push_back(makeComposedAffineApply(
          rewriter, loc, dim0 + dim1, {writeOp.getIndices()[i], offsetVal}));
    }

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
    if (!areAllRankReducedLeadingDim(insertOp,
                                     insertOp.getSourceType().getRank()))
      return rewriter.notifyMatchFailure(insertOp, "not leading rank reduced");

    unsigned writeTensorRank = writeOp.getSource().getType().getRank();
    unsigned writeReducedRank = writeOp.getLeadingShapedRank();

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
