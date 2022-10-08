//===- ViewLikeInterfaceUtils.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/ViewLikeInterfaceUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"

using namespace mlir;

LogicalResult mlir::mergeOffsetsSizesAndStrides(
    OpBuilder &builder, Location loc, ArrayRef<OpFoldResult> producerOffsets,
    ArrayRef<OpFoldResult> producerSizes,
    ArrayRef<OpFoldResult> producerStrides,
    const llvm::SmallBitVector &droppedProducerDims,
    ArrayRef<OpFoldResult> consumerOffsets,
    ArrayRef<OpFoldResult> consumerSizes,
    ArrayRef<OpFoldResult> consumerStrides,
    SmallVector<OpFoldResult> &combinedOffsets,
    SmallVector<OpFoldResult> &combinedSizes,
    SmallVector<OpFoldResult> &combinedStrides) {
  combinedOffsets.resize(producerOffsets.size());
  combinedSizes.resize(producerOffsets.size());
  combinedStrides.resize(producerOffsets.size());

  AffineExpr s0, s1, s2;
  bindSymbols(builder.getContext(), s0, s1, s2);

  unsigned consumerPos = 0;
  for (auto i : llvm::seq<unsigned>(0, producerOffsets.size())) {
    if (droppedProducerDims.test(i)) {
      // For dropped dims, get the values from the producer.
      combinedOffsets[i] = producerOffsets[i];
      combinedSizes[i] = producerSizes[i];
      combinedStrides[i] = producerStrides[i];
      continue;
    }
    SmallVector<OpFoldResult> offsetSymbols, strideSymbols;
    // The combined offset is computed as
    //    producer_offset + consumer_offset * producer_strides.
    combinedOffsets[i] = makeComposedFoldedAffineApply(
        builder, loc, s0 * s1 + s2,
        {consumerOffsets[consumerPos], producerStrides[i], producerOffsets[i]});
    combinedSizes[i] = consumerSizes[consumerPos];
    // The combined stride is computed as
    //    consumer_stride * producer_stride.
    combinedStrides[i] = makeComposedFoldedAffineApply(
        builder, loc, s0 * s1,
        {consumerStrides[consumerPos], producerStrides[i]});

    consumerPos++;
  }
  return success();
}

LogicalResult mlir::mergeOffsetsSizesAndStrides(
    OpBuilder &builder, Location loc, OffsetSizeAndStrideOpInterface producer,
    OffsetSizeAndStrideOpInterface consumer,
    const llvm::SmallBitVector &droppedProducerDims,
    SmallVector<OpFoldResult> &combinedOffsets,
    SmallVector<OpFoldResult> &combinedSizes,
    SmallVector<OpFoldResult> &combinedStrides) {
  SmallVector<OpFoldResult> consumerOffsets = consumer.getMixedOffsets();
  SmallVector<OpFoldResult> consumerSizes = consumer.getMixedSizes();
  SmallVector<OpFoldResult> consumerStrides = consumer.getMixedStrides();
  SmallVector<OpFoldResult> producerOffsets = producer.getMixedOffsets();
  SmallVector<OpFoldResult> producerSizes = producer.getMixedSizes();
  SmallVector<OpFoldResult> producerStrides = producer.getMixedStrides();
  return mergeOffsetsSizesAndStrides(
      builder, loc, producerOffsets, producerSizes, producerStrides,
      droppedProducerDims, consumerOffsets, consumerSizes, consumerStrides,
      combinedOffsets, combinedSizes, combinedStrides);
}

bool mlir::areDisjointRanges(ArrayRef<OpFoldResult> aOffsets,
                             ArrayRef<OpFoldResult> aSizes,
                             ArrayRef<OpFoldResult> bOffsets,
                             ArrayRef<OpFoldResult> bSizes) {
  assert(llvm::all_equal(
      {aOffsets.size(), aSizes.size(), bOffsets.size(), bSizes.size()}));

  for (const auto &t : llvm::zip(aOffsets, aSizes, bOffsets, bSizes)) {
    auto [aBeginVal, aSizeVal, bBeginVal, bSizeVal] = t;
    Optional<int64_t> aBegin = getConstantIntValue(aBeginVal);
    Optional<int64_t> aSize = getConstantIntValue(aSizeVal);
    Optional<int64_t> bBegin = getConstantIntValue(bBeginVal);
    Optional<int64_t> bSize = getConstantIntValue(bSizeVal);

    // If there are dynamic offsets/sizes, we cannot prove this dimension is
    // disjoint. Look at other dimensions.
    if (!aBegin || !aSize || !bBegin || !bSize)
      continue;

    int aEnd = *aBegin + *aSize;
    int bEnd = *bBegin + *bSize;
    // As long as one dimension is disjoint, the whole slices are disjoint.
    if (aEnd <= *bBegin || bEnd <= *aBegin)
      return true;
  }
  return false;
}

bool mlir::areDisjointSlices(OffsetSizeAndStrideOpInterface aSlice,
                             OffsetSizeAndStrideOpInterface bSlice) {
  SmallVector<OpFoldResult> aOffsets = aSlice.getMixedOffsets();
  SmallVector<OpFoldResult> bOffsets = bSlice.getMixedOffsets();
  SmallVector<OpFoldResult> aSizes = aSlice.getMixedSizes();
  SmallVector<OpFoldResult> bSizes = bSlice.getMixedSizes();
  SmallVector<OpFoldResult> aStrides = aSlice.getMixedStrides();
  SmallVector<OpFoldResult> bStrides = bSlice.getMixedStrides();

  // For simplicity only look at stride 1 cases for now.
  auto hasAllOnes = [](ArrayRef<OpFoldResult> strides) {
    return llvm::all_of(strides, [](::mlir::OpFoldResult ofr) {
      return getConstantIntValue(ofr) == static_cast<int64_t>(1);
    });
  };
  if (!hasAllOnes(aStrides) || !hasAllOnes(bStrides))
    return false;

  return areDisjointRanges(aOffsets, aSizes, bOffsets, bSizes);
}
