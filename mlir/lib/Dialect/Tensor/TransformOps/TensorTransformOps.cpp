
//===- TensorTransformOps.cpp - Implementation of tensor transform ops ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tensor/TransformOps/TensorTransformOps.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// TrackingListener
//===----------------------------------------------------------------------===//

void tensor::TrackingListener::notifyOperationRemoved(Operation *op) {
  // TODO: Walk can be removed when D144193 has landed.
  op->walk([&](Operation *op) {
    // Keep set of new ops up-to-date.
    auto it = newOps.find(op->getName());
    if (it != newOps.end())
      it->second.erase(op);
    // Remove mappings for result values.
    for (OpResult value : op->getResults())
      (void)replacePayloadValue(value, nullptr);
    // Remove mapping for op.
    (void)replacePayloadOp(op, nullptr);
  });
}

void tensor::TrackingListener::notifyOperationReplaced(Operation *op,
                                                       ValueRange newValues) {
  assert(op->getNumResults() == newValues.size() &&
         "invalid number of replacement values");

  // Replace value handles.
  for (auto [oldValue, newValue] : llvm::zip(op->getResults(), newValues))
    (void)replacePayloadValue(oldValue, newValue);

  // Try to replace op handle.
  SmallVector<Value> values(newValues.begin(), newValues.end());
  // Consider only ops that define all replacement values.
  while (llvm::all_equal(
      llvm::map_range(values, [](Value v) { return v.getDefiningOp(); }))) {
    // The defining ops is a replacement if it is the same type of op and was
    // newly created during the rewrite.
    Operation *defOp = values.front().getDefiningOp();
    if (!defOp)
      return;
    if (defOp->getName() == op->getName()) {
      auto it = newOps.find(op->getName());
      if (it != newOps.end()) {
        if (it->second.contains(defOp)) {
          (void)replacePayloadOp(op, defOp);
          return;
        }
      }
    }

    // Query DestinationStyleOpInterface to find a suitable replacement.
    auto dstOp = dyn_cast<DestinationStyleOpInterface>(defOp);
    if (!dstOp)
      return;
    SmallVector<Value> tiedOperands;
    for (int i = 0; i < values.size(); ++i) {
      OpResult opResult = values[i].cast<OpResult>();
      // Stop lookup if the value has no tied/init operand.
      if (!dstOp.hasTiedOpOperand(opResult))
        return;
      values[i] = dstOp.getTiedOpOperand(opResult)->get();
    }
  }
}
