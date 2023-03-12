//===- ValueBoundsOpInterface.h - Value Bounds ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LINALG_TRANSFORMS_VALUEBOUNDSOPINTERFACE_H_
#define MLIR_DIALECT_LINALG_TRANSFORMS_VALUEBOUNDSOPINTERFACE_H_

#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "llvm/ADT/SetVector.h"

namespace mlir {
namespace linalg {

/// A helper class to be used with `ValueBoundsOpInterface`. This class stores a
/// constraint system and mapping of columns to values/shape dimensions.
///
/// Note: This class maintains its own mapping to SSA values; no SSA values are
/// mapped in the underlying `FlatAffineValueConstraints`. This is because not
/// only SSA values but also shape dimensions of SSA values must be mapped.
class ValueBoundsConstraintSet {
public:
  /// Reify a bound for the given index-typed value or shape dimension size in
  /// terms of the owning op's operands. LB and EQ bounds are closed, UB bounds
  /// are open.
  static FailureOr<OpFoldResult>
  reifyBound(OpBuilder &b, Location loc,
             presburger::IntegerPolyhedron::BoundType type, Value value,
             int64_t dim = kIndexValue);

  /// Reify a bound for the given index-typed value or shape dimension size in
  /// terms of SSA values for which `stopCondition` is met. LB and EQ bounds are
  /// closed, UB bounds are open.
  static FailureOr<OpFoldResult>
  reifyBound(OpBuilder &b, Location loc,
             presburger::IntegerPolyhedron::BoundType type, Value value,
             int64_t dim, function_ref<bool(Value)> stopCondition);

  /// Dimension indentifier to indicate a value is index-typed.
  static const int64_t kIndexValue = -1;

  /// Bound the given index-typed value by the given expression.
  void addBound(presburger::IntegerPolyhedron::BoundType type, Value value,
                AffineExpr expr);

  /// Bound the the given shaped value dimension by the given expression.
  void addBound(presburger::IntegerPolyhedron::BoundType type, Value value,
                int64_t dim, AffineExpr expr);

  /// Bound the given index-typed value by a constant or SSA value.
  void addBound(presburger::IntegerPolyhedron::BoundType type, Value value,
                OpFoldResult ofr);

  /// Bound the the given shaped value dimension by a constant or SSA value.
  void addBound(presburger::IntegerPolyhedron::BoundType type, Value value,
                int64_t dim, OpFoldResult ofr);

  /// Return an expression that represents the given index-typed value or shaped
  /// value dimension. If this value/dimension was not used so far, it is added
  /// to the worklist.
  AffineExpr getExpr(Value value, int64_t dim = kIndexValue);

  /// Return an expression that represents a constant or index-typed SSA value.
  /// In case of a value, if this value was not used so far, it is added to the
  /// worklist.
  AffineExpr getExpr(OpFoldResult ofr);

  /// Return an expression that represents a constant.
  AffineExpr getExpr(int64_t val);

private:
  using ValueDim = std::pair<Value, int64_t>;
  ValueBoundsConstraintSet(ValueDim valueDim);

  /// Iteratively process all elements on the worklist until an index-typed
  /// value or shaped value meets `stopCondition`. Such values are not processed
  /// any further.
  void processWorklist(function_ref<bool(Value)> stopCondition);

  /// Bound the given column in the underlying constraint set by the given
  /// expression.
  void addBound(presburger::IntegerPolyhedron::BoundType type, int64_t pos,
                AffineExpr expr);

  /// Return the column position of the given value/dimension. Asserts that the
  /// value/dimension exists in the constraint set.
  int64_t getPos(Value value, int64_t dim = kIndexValue) const;

  /// Insert a value/dimension into the constraint set. If `isSymbol` is set to
  /// "false", a dimension is added.
  ///
  /// Note: There are certain affine restrictions wrt. dimensions. E.g., they
  /// cannot be multiplied. Furthermore, bounds can only be queried for
  /// dimensions but not for symbols.
  int64_t insert(ValueDim valueDim, bool isSymbol = true);

  /// Project out the given column in the constraint set.
  void projectOut(int64_t pos);

  /// Mapping of columns to values/shape dimensions.
  SmallVector<ValueDim> positionToValueDim;
  /// Reverse mapping of values/shape dimensions to columns.
  DenseMap<ValueDim, int64_t> valueDimToPosition;

  /// Worklist of values/shape dimensions that have not been processed yet.
  SetVector<int64_t> worklist;

  /// Constraint system of equalities and inequalities.
  FlatAffineValueConstraints cstr;

  /// Builder for constructing affine expressions.
  Builder builder;
};
} // namespace linalg
} // namespace mlir

#include "mlir/Dialect/Linalg/Transforms/ValueBoundsOpInterface.h.inc"

namespace mlir {
namespace linalg {

/// Default implementation for destination style ops: Tied OpResults and
/// OpOperands have the same type.
template <typename ConcreteOp>
struct DstValueBoundsOpInterfaceExternalModel
    : public ValueBoundsOpInterface::ExternalModel<
          DstValueBoundsOpInterfaceExternalModel<ConcreteOp>, ConcreteOp> {
  void populateBoundsForShapedValueDim(Operation *op, Value value, int64_t dim,
                                       ValueBoundsConstraintSet &cstr) const {
    auto dstOp = cast<DestinationStyleOpInterface>(op);
    assert(value.getDefiningOp() == dstOp);

    Value tiedOperand = dstOp.getTiedOpOperand(value.cast<OpResult>())->get();
    cstr.addBound(presburger::IntegerPolyhedron::BoundType::EQ, value, dim,
                  cstr.getExpr(tiedOperand, dim));
  }
};

} // namespace linalg
} // namespace mlir

#endif // MLIR_DIALECT_LINALG_TRANSFORMS_VALUEBOUNDSOPINTERFACE_H_
