//===- ValueBoundsOpInterface.h - Value Bounds ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_VALUEBOUNDSOPINTERFACE_H_
#define MLIR_INTERFACES_VALUEBOUNDSOPINTERFACE_H_

#include "mlir/Analysis/FlatValueConstraints.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "llvm/ADT/SetVector.h"

namespace mlir {

using ValueDimList = SmallVector<std::pair<Value, std::optional<int64_t>>>;

/// A helper class to be used with `ValueBoundsOpInterface`. This class stores a
/// constraint system and mapping of constrained variables to index-typed
/// values or dimension sizes of shaped values.
///
/// Interface implementations of `ValueBoundsOpInterface` use `addBounds` to
/// insert constraints about their results and/or region block arguments into
/// the constraint set in the form of an AffineExpr. When a bound should be
/// expressed in terms of another value/dimension, `getExpr` can be used to
/// retrieve an AffineExpr that represents the specified value/dimension.
///
/// When a value/dimension is retrieved for the first time through `getExpr`,
/// it is added to an internal worklist. See `computeBound` for more details.
///
/// Note: Any modification of the IR invalides the data stored in this class.
class ValueBoundsConstraintSet {
public:
  /// Compute a bound for the given index-typed value or shape dimension size.
  /// The computed bound is stored in `resultMap`. The operands of the bound are
  /// stored in `mapOperands`. An operand is either an index-type SSA value
  /// or a shaped value and a dimension.
  ///
  /// `dim` must be `nullopt` if and only if `value` is index-typed. The bound
  /// is computed in terms of values for which `stopCondition` evaluates to
  /// "true". To that end, the backward slice (reverse use-def chain) of the
  /// given value is visited in a worklist-driven manner and the constraint set
  /// is populated according to `ValueBoundsOpInterface` for each visited value.
  static LogicalResult computeBound(AffineMap &resultMap,
                                    ValueDimList &mapOperands, Location loc,
                                    presburger::BoundType type, Value value,
                                    std::optional<int64_t> dim,
                                    function_ref<bool(Value)> stopCondition);

  /// Bound the given index-typed value by the given expression.
  void addBound(presburger::BoundType type, Value value, AffineExpr expr);

  /// Bound the the given shaped value dimension by the given expression.
  void addBound(presburger::BoundType type, Value value, int64_t dim,
                AffineExpr expr);

  /// Bound the given index-typed value by a constant or SSA value.
  void addBound(presburger::BoundType type, Value value, OpFoldResult ofr);

  /// Bound the the given shaped value dimension by a constant or SSA value.
  void addBound(presburger::BoundType type, Value value, int64_t dim,
                OpFoldResult ofr);

  /// Return an expression that represents the given index-typed value or shaped
  /// value dimension. If this value/dimension was not used so far, it is added
  /// to the worklist.
  ///
  /// `dim` must be `nullopt` if and only if the given value is of index type.
  AffineExpr getExpr(Value value, std::optional<int64_t> dim = std::nullopt);

  /// Return an expression that represents a constant or index-typed SSA value.
  /// In case of a value, if this value was not used so far, it is added to the
  /// worklist.
  AffineExpr getExpr(OpFoldResult ofr);

  /// Return an expression that represents a constant.
  AffineExpr getExpr(int64_t constant);

protected:
  /// Dimension identifier to indicate a value is index-typed.
  static constexpr int64_t kIndexValue = -1;

  using ValueDim = std::pair<Value, int64_t>;
  ValueBoundsConstraintSet(ValueDim valueDim);

  /// Iteratively process all elements on the worklist until an index-typed
  /// value or shaped value meets `stopCondition`. Such values are not processed
  /// any further.
  void processWorklist(function_ref<bool(Value)> stopCondition);

  /// Bound the given column in the underlying constraint set by the given
  /// expression.
  void addBound(presburger::BoundType type, int64_t pos, AffineExpr expr);

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
  FlatConstraints cstr;

  /// Builder for constructing affine expressions.
  Builder builder;
};

} // namespace mlir

#include "mlir/Interfaces/ValueBoundsOpInterface.h.inc"

namespace mlir {

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
    cstr.addBound(presburger::BoundType::EQ, value, dim,
                  cstr.getExpr(tiedOperand, dim));
  }
};

} // namespace mlir

#endif // MLIR_INTERFACES_VALUEBOUNDSOPINTERFACE_H_
