//===- ViewLikeInterfaceUtils.h ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_AFFINE_VIEWLIKEINTERFACEUTILS_H
#define MLIR_DIALECT_AFFINE_VIEWLIKEINTERFACEUTILS_H

#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

namespace mlir {

/// Fills the `combinedOffsets`, `combinedSizes` and `combinedStrides` to use
/// when combining a producer slice **into** a consumer slice.
///
/// This function performs the following computation:
/// - Combined offsets = producer_offsets * consumer_strides + consumer_offsets
/// - Combined sizes = consumer_sizes
/// - Combined strides = producer_strides * consumer_strides
LogicalResult
mergeOffsetsSizesAndStrides(OpBuilder &builder, Location loc,
                            ArrayRef<OpFoldResult> producerOffsets,
                            ArrayRef<OpFoldResult> producerSizes,
                            ArrayRef<OpFoldResult> producerStrides,
                            const llvm::SmallBitVector &droppedProducerDims,
                            ArrayRef<OpFoldResult> consumerOffsets,
                            ArrayRef<OpFoldResult> consumerSizes,
                            ArrayRef<OpFoldResult> consumerStrides,
                            SmallVector<OpFoldResult> &combinedOffsets,
                            SmallVector<OpFoldResult> &combinedSizes,
                            SmallVector<OpFoldResult> &combinedStrides);

/// Fills the `combinedOffsets`, `combinedSizes` and `combinedStrides` to use
/// when combining a `producer` slice op **into** a `consumer` slice op.
LogicalResult
mergeOffsetsSizesAndStrides(OpBuilder &builder, Location loc,
                            OffsetSizeAndStrideOpInterface producer,
                            OffsetSizeAndStrideOpInterface consumer,
                            const llvm::SmallBitVector &droppedProducerDims,
                            SmallVector<OpFoldResult> &combinedOffsets,
                            SmallVector<OpFoldResult> &combinedSizes,
                            SmallVector<OpFoldResult> &combinedStrides);

/// Returns true if the given two n-D ranges can be proven as disjoint.
/// Returns false otherwise.
///
/// This function assumes all input arrays to have the same size.
bool areDisjointRanges(ArrayRef<OpFoldResult> aOffsets,
                       ArrayRef<OpFoldResult> aSizes,
                       ArrayRef<OpFoldResult> bOffsets,
                       ArrayRef<OpFoldResult> bSizes);

/// Returns true if the given two slices can be proven as disjoint. Returns
/// false otherwise.
///
/// This function assumes the two slices have the same rank.
bool areDisjointSlices(OffsetSizeAndStrideOpInterface aSlice,
                       OffsetSizeAndStrideOpInterface bSlice);

} // namespace mlir

#endif // MLIR_DIALECT_AFFINE_VIEWLIKEINTERFACEUTILS_H
