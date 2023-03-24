//===- TensorTransformOps.h - Tensor transformation ops ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TENSOR_TRANSFORMOPS_TENSORTRANSFORMOPS_H
#define MLIR_DIALECT_TENSOR_TRANSFORMOPS_TENSORTRANSFORMOPS_H

#include "mlir/Dialect/PDL/IR/PDLTypes.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace tensor {

/// A specialized TrackingListener for transform ops that operate on tensor IR.
/// This listener utilizes the DestinationStyleOpInterface to find replacement
/// ops. Only newly-created ops are considered as replacement ops.
class TrackingListener : public RewriterBase::Listener,
                         public transform::TransformState::Extension {
public:
  explicit TrackingListener(transform::TransformState &state)
      : transform::TransformState::Extension(state) {}

  void notifyOperationInserted(Operation *op) override;

  void notifyOperationReplaced(Operation *op, ValueRange newValues) override;

  void notifyOperationRemoved(Operation *op) override;

private:
  /// Ops that were newly created during the transform.
  DenseMap<OperationName, DenseSet<Operation *>> newOps;
};

} // namespace tensor
} // namespace mlir

#endif // MLIR_DIALECT_TENSOR_TRANSFORMOPS_TENSORTRANSFORMOPS_H
