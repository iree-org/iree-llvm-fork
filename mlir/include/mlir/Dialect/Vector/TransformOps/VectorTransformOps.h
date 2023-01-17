//===- VectorTransformOps.h - Vector transform ops --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_VECTOR_TRANSFORMOPS_VECTORTRANSFORMOPS_H
#define MLIR_DIALECT_VECTOR_TRANSFORMOPS_VECTORTRANSFORMOPS_H

#include "mlir/Dialect/PDL/IR/PDLTypes.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/IR/OpImplementation.h"

namespace mlir {
namespace vector {
class VectorOp;
struct LowerVectorsOptions;
} // namespace vector
} // namespace mlir

//===----------------------------------------------------------------------===//
// Vector Transform Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/Vector/TransformOps/VectorTransformOps.h.inc"

namespace mlir {
class DialectRegistry;

namespace vector {
void registerTransformDialectExtension(DialectRegistry &registry);

/// Helper structure used to hold the different options of LowerVectorsOp.
struct LowerVectorsOptions : public VectorTransformsOptions {
  // Have the default values match the LowerVectorsOp values in the td file.
  LowerVectorsOptions() : VectorTransformsOptions() {
    setVectorTransformsOptions(VectorContractLowering::OuterProduct);
    setVectorMultiReductionLowering(
        VectorMultiReductionLowering::InnerParallel);
    setVectorTransposeLowering(VectorTransposeLowering::EltWise);
    setVectorTransferSplit(VectorTransferSplit::LinalgCopy);
  }

  bool transposeAVX2Lowering = false;
  LowerVectorsOptions &setTransposeAVX2Lowering(bool opt) {
    transposeAVX2Lowering = opt;
    return *this;
  }

  bool unrollVectorTransfers = true;
  LowerVectorsOptions &setUnrollVectorTransfers(bool opt) {
    unrollVectorTransfers = opt;
    return *this;
  }
};
} // namespace vector
} // namespace mlir

#endif // MLIR_DIALECT_VECTOR_TRANSFORMOPS_VECTORTRANSFORMOPS_H
