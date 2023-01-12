//===- VectorMaskingUtils.h - Vector masking utilitites -----*- C++------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_VECTOR_TRANSFORMS_VECTORMASKINGUTILS_H_
#define MLIR_DIALECT_VECTOR_TRANSFORMS_VECTORMASKINGUTILS_H_

#include "mlir/Dialect/Vector/IR/VectorOps.h"

namespace mlir {
namespace vector {
namespace utils {

/// Create the vector.yield-ended region of a vector.mask op with `maskableOp`
/// as masked operation.
static void createMaskOpRegion(OpBuilder &builder, Operation *maskableOp) {
  assert(maskableOp->getBlock() && "MaskableOp must be inserted into a block");
  Block *insBlock = builder.getInsertionBlock();
  // Create a block and move the op to that block.
  insBlock->getOperations().splice(
      insBlock->begin(), maskableOp->getBlock()->getOperations(), maskableOp);
  builder.create<YieldOp>(maskableOp->getLoc(), maskableOp->getResults());
}

/// Creates a vector.mask operation around a maskable operation to be masked.
/// Returns the vector.mask operation if the mask provided is valid. Otherwise,
/// returns the maskable operation itself.
static Operation *maskOperation(RewriterBase &rewriter, Operation *maskableOp,
                                Value mask) {
  if (!mask)
    return maskableOp;

  return maskableOp->getResults().empty()
             ? rewriter.create<MaskOp>(maskableOp->getLoc(), mask, maskableOp,
                                       createMaskOpRegion)
             : rewriter.create<MaskOp>(maskableOp->getLoc(),
                                       maskableOp->getResultTypes().front(),
                                       mask, maskableOp, createMaskOpRegion);
}

} // namespace utils
} // namespace vector
} // namespace mlir

#endif // MLIR_DIALECT_VECTOR_TRANSFORMS_VECTORMASKINGUTILS_H_
