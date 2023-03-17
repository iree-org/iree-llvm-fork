//===- ValueBoundsOpInterfaceImpl.cpp - Impl. of ValueBoundsOpInterface ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SCF/IR/ValueBoundsOpInterfaceImpl.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"

using namespace mlir;
using presburger::BoundType;

namespace mlir {
namespace scf {
namespace {

struct ForOpInterface
    : public ValueBoundsOpInterface::ExternalModel<ForOpInterface, ForOp> {
  void populateBoundsForIndexValue(Operation *op, Value value,
                                   ValueBoundsConstraintSet &cstr) const {
    auto forOp = cast<ForOp>(op);

    if (value == forOp.getInductionVar()) {
      // TODO: Take into account step size.
      cstr.addBound(BoundType::LB, value, cstr.getExpr(forOp.getLowerBound()));
      cstr.addBound(BoundType::UB, value, cstr.getExpr(forOp.getUpperBound()));
      return;
    }

    // `value` is an iter_arg or an OpResult.
    int64_t iterArgIdx;
    if (auto iterArg = value.dyn_cast<BlockArgument>()) {
      iterArgIdx = iterArg.getArgNumber() - forOp.getNumInductionVars();
    } else {
      iterArgIdx = value.cast<OpResult>().getResultNumber();
    }

    // An EQ constraint can be added if the yielded value is the same value as
    // the corresponding block argument.
    assert(forOp.getLoopBody().hasOneBlock() &&
           "multiple blocks not supported");
    Value yieldedValue =
        cast<scf::YieldOp>(forOp.getLoopBody().front().getTerminator())
            .getOperand(iterArgIdx);
    Value iterArg = forOp.getRegionIterArg(iterArgIdx);
    Value initArg = forOp.getInitArgs()[iterArgIdx];

    // Compute EQ bound for yielded value.
    AffineMap bound;
    ValueDimList boundOperands;
    LogicalResult status = ValueBoundsConstraintSet::computeBound(
        bound, boundOperands, BoundType::EQ, yieldedValue,
        /*dim=*/std::nullopt, {{iterArg, std::nullopt}});
    if (failed(status))
      return;
    if (bound.getNumResults() != 1)
      return;

    // Check if computed bound equals the corresponding iter_arg.
    Value singleValue = nullptr;
    if (auto dimExpr = bound.getResult(0).dyn_cast<AffineDimExpr>()) {
      int64_t idx = dimExpr.getPosition();
      if (!boundOperands[idx].second.has_value())
        singleValue = boundOperands[idx].first;
    } else if (auto symExpr = bound.getResult(0).dyn_cast<AffineSymbolExpr>()) {
      int64_t idx = symExpr.getPosition() + bound.getNumDims();
      if (!boundOperands[idx].second.has_value())
        singleValue = boundOperands[idx].first;
    }
    if (singleValue == iterArg)
      cstr.addBound(BoundType::EQ, value, cstr.getExpr(initArg));
  }

  void populateBoundsForShapedValueDim(Operation *op, Value value, int64_t dim,
                                       ValueBoundsConstraintSet &cstr) const {
    auto forOp = cast<ForOp>(op);

    // `value` is an iter_arg or an OpResult.
    int64_t iterArgIdx;
    if (auto iterArg = value.dyn_cast<BlockArgument>()) {
      iterArgIdx = iterArg.getArgNumber() - forOp.getNumInductionVars();
    } else {
      iterArgIdx = value.cast<OpResult>().getResultNumber();
    }

    // An EQ constraint can be added if the dimension size of the yielded value
    // equals the dimensions size of the corresponding block argument.
    assert(forOp.getLoopBody().hasOneBlock() &&
           "multiple blocks not supported");
    Value yieldedValue =
        cast<scf::YieldOp>(forOp.getLoopBody().front().getTerminator())
            .getOperand(iterArgIdx);
    Value iterArg = forOp.getRegionIterArg(iterArgIdx);
    Value initArg = forOp.getInitArgs()[iterArgIdx];

    // Compute EQ bound for yielded value.
    AffineMap bound;
    ValueDimList boundOperands;
    LogicalResult status = ValueBoundsConstraintSet::computeBound(
        bound, boundOperands, BoundType::EQ, yieldedValue, dim,
        {{iterArg, dim}});
    if (failed(status))
      return;
    if (bound.getNumResults() != 1)
      return;

    // Check if computed bound equals the corresponding iter_arg.
    Value singleValue = nullptr;
    int64_t singleDim = -1;
    if (auto dimExpr = bound.getResult(0).dyn_cast<AffineDimExpr>()) {
      int64_t idx = dimExpr.getPosition();
      if (boundOperands[idx].second.has_value()) {
        singleValue = boundOperands[idx].first;
        singleDim = *boundOperands[idx].second;
      }
    } else if (auto symExpr = bound.getResult(0).dyn_cast<AffineSymbolExpr>()) {
      int64_t idx = symExpr.getPosition() + bound.getNumDims();
      if (boundOperands[idx].second.has_value()) {
        singleValue = boundOperands[idx].first;
        singleDim = *boundOperands[idx].second;
      }
    }
    if (singleValue == iterArg && singleDim == dim)
      cstr.addBound(BoundType::EQ, value, dim, cstr.getExpr(initArg, dim));
  }
};

} // namespace
} // namespace scf
} // namespace mlir

void mlir::scf::registerValueBoundsOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, scf::SCFDialect *dialect) {
    scf::ForOp::attachInterface<scf::ForOpInterface>(*ctx);
  });
}
