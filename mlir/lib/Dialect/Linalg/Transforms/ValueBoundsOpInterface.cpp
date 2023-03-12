//===- ValueBoundsOpInterface.cpp - Value Bounds  -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/Transforms/ValueBoundsOpInterface.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;
using namespace mlir::linalg;
using presburger::IntegerPolyhedron;

namespace mlir {
#include "mlir/Dialect/Linalg/Transforms/ValueBoundsOpInterface.cpp.inc"
} // namespace mlir

ValueBoundsConstraintSet::ValueBoundsConstraintSet(ValueDim valueDim)
    : builder(valueDim.first.getContext()) {
  insert(valueDim, /*isSymbol=*/false);
}

#ifndef NDEBUG
static void assertValidValueDim(Value value, int64_t dim) {
  if (value.getType().isIndex()) {
    assert(dim == ValueBoundsConstraintSet::kIndexValue && "invalid dim value");
  } else if (auto shapedType = value.getType().dyn_cast<ShapedType>()) {
    assert(dim >= 0 && "invalid dim value");
    if (shapedType.hasRank())
      assert(dim < shapedType.getRank() && "invalid dim value");
  } else {
    llvm_unreachable("unsupported type");
  }
}
#endif // NDEBUG

void ValueBoundsConstraintSet::addBound(
    presburger::IntegerPolyhedron::BoundType type, int64_t pos,
    AffineExpr expr) {
  LogicalResult status = cstr.addBound(
      type, pos,
      AffineMap::get(cstr.getNumDimVars(), cstr.getNumSymbolVars(), expr));
  (void)status;
  assert(succeeded(status) && "failed to add bound to constraint system");
}

void ValueBoundsConstraintSet::addBound(
    presburger::IntegerPolyhedron::BoundType type, Value value,
    AffineExpr expr) {
  assert(value.getType().isIndex() && "expected index type");
  addBound(type, getPos(value), expr);
}

void ValueBoundsConstraintSet::addBound(
    presburger::IntegerPolyhedron::BoundType type, Value value, int64_t dim,
    AffineExpr expr) {
#ifndef NDEBUG
  assertValidValueDim(value, dim);
#endif // NDEBUG
  addBound(type, getPos(value, dim), expr);
}

void ValueBoundsConstraintSet::addBound(
    presburger::IntegerPolyhedron::BoundType type, Value value,
    OpFoldResult ofr) {
  assert(value.getType().isIndex() && "expected index type");
  addBound(type, getPos(value), getExpr(ofr));
}

void ValueBoundsConstraintSet::addBound(
    presburger::IntegerPolyhedron::BoundType type, Value value, int64_t dim,
    OpFoldResult ofr) {
#ifndef NDEBUG
  assertValidValueDim(value, dim);
#endif // NDEBUG
  addBound(type, getPos(value, dim), getExpr(ofr));
}

AffineExpr ValueBoundsConstraintSet::getExpr(Value value, int64_t dim) {
#ifndef NDEBUG
  assertValidValueDim(value, dim);
#endif // NDEBUG

  auto shapedType = value.getType().dyn_cast<ShapedType>();
  if (shapedType) {
    // Static dimension: return constant directly.
    if (shapedType.hasRank() && !shapedType.isDynamicDim(dim))
      return builder.getAffineConstantExpr(shapedType.getDimSize(dim));
  } else {
    // Constant index value: return directly.
    if (auto constInt = getConstantIntValue(value))
      return builder.getAffineConstantExpr(*constInt);
  }

  // Dynamic value: add to constraint set.
  ValueDim valueDim = std::make_pair(value, dim);
  if (valueDimToPosition.find(valueDim) == valueDimToPosition.end())
    (void)insert(valueDim);
  int64_t pos = getPos(value, dim);
  return pos < cstr.getNumDimVars()
             ? builder.getAffineDimExpr(pos)
             : builder.getAffineSymbolExpr(pos - cstr.getNumDimVars());
}

AffineExpr ValueBoundsConstraintSet::getExpr(OpFoldResult ofr) {
  if (Value value = ofr.dyn_cast<Value>())
    return getExpr(value, /*dim=*/kIndexValue);
  auto constInt = getConstantIntValue(ofr);
  assert(constInt.has_value() && "expected Integer constant");
  return builder.getAffineConstantExpr(*constInt);
}

AffineExpr ValueBoundsConstraintSet::getExpr(int64_t val) {
  return builder.getAffineConstantExpr(val);
}

int64_t ValueBoundsConstraintSet::insert(ValueDim valueDim, bool isSymbol) {
  assert((valueDimToPosition.find(valueDim) == valueDimToPosition.end()) &&
         "already mapped");
  int64_t pos = isSymbol ? cstr.appendSymbolVar() : cstr.appendDimVar();
  positionToValueDim.insert(positionToValueDim.begin() + pos, valueDim);
  // Update reverse mapping.
  for (int64_t i = pos; i < positionToValueDim.size(); ++i)
    valueDimToPosition[positionToValueDim[i]] = i;

  worklist.insert(pos);
  return pos;
}

int64_t ValueBoundsConstraintSet::getPos(Value value, int64_t dim) const {
#ifndef NDEBUG
  assertValidValueDim(value, dim);
#endif // NDEBUG

  auto it = valueDimToPosition.find(std::make_pair(value, dim));
  assert(it != valueDimToPosition.end() && "expected mapped entry");
  return it->second;
}

static Operation *getOwnerOfValue(Value value) {
  if (auto bbArg = value.dyn_cast<BlockArgument>())
    return bbArg.getOwner()->getParentOp();
  return value.getDefiningOp();
}

void ValueBoundsConstraintSet::processWorklist(
    function_ref<bool(Value)> stopCondition) {
  while (!worklist.empty()) {
    int64_t pos = worklist.pop_back_val();
    ValueDim valueDim = positionToValueDim[pos];
    Value value = valueDim.first;
    int64_t dim = valueDim.second;

    // Check for static dim size.
    if (dim != kIndexValue) {
      auto shapedType = value.getType().cast<ShapedType>();
      if (shapedType.hasRank() && !shapedType.isDynamicDim(dim)) {
        addBound(IntegerPolyhedron::BoundType::EQ, value, dim,
                 builder.getAffineConstantExpr(shapedType.getDimSize(dim)));
        continue;
      }
    }

    // Do not process any further if the stop condition is met.
    if (stopCondition(value))
      continue;

    // Query `ValueBoundsOpInterface` for constraints. New items may be added to
    // the worklist.
    auto valueBoundsOp =
        dyn_cast<ValueBoundsOpInterface>(getOwnerOfValue(value));
    if (!valueBoundsOp)
      continue;
    if (dim == kIndexValue) {
      valueBoundsOp.populateBoundsForIndexValue(value, *this);
    } else {
      valueBoundsOp.populateBoundsForShapedValueDim(value, dim, *this);
    }
  }
}

void ValueBoundsConstraintSet::projectOut(int64_t pos) {
  cstr.projectOut(pos);
  bool erased = valueDimToPosition.erase(positionToValueDim[pos]);
  assert(erased && "inconsistent reverse mapping");
  positionToValueDim.erase(positionToValueDim.begin() + pos);
  // Update reverse mapping.
  for (int64_t i = pos; i < positionToValueDim.size(); ++i)
    valueDimToPosition[positionToValueDim[i]] = i;
}

FailureOr<OpFoldResult> ValueBoundsConstraintSet::reifyBound(
    OpBuilder &b, Location loc, presburger::IntegerPolyhedron::BoundType type,
    Value value, int64_t dim) {
  auto stopCondition = [&](Value v) {
    // Reify in terms of SSA values that are different from `value`.
    return v != value;
  };
  return ValueBoundsConstraintSet::reifyBound(b, loc, type, value, dim,
                                              stopCondition);
}

FailureOr<OpFoldResult> ValueBoundsConstraintSet::reifyBound(
    OpBuilder &b, Location loc, presburger::IntegerPolyhedron::BoundType type,
    Value value, int64_t dim, function_ref<bool(Value)> stopCondition) {
#ifndef NDEBUG
  assertValidValueDim(value, dim);
#endif // NDEBUG

  // Process the backward slice of `value` (i.e., reverse use-def chain) until
  // `stopCondition` is met.
  ValueBoundsConstraintSet cstr(std::make_pair(value, dim));
  int64_t pos = cstr.getPos(value, dim);
  cstr.processWorklist(stopCondition);

  // Project out all positions (apart from `pos`) that do not match the stop
  // condition.
  int64_t nextPos = 0;
  while (nextPos < cstr.positionToValueDim.size()) {
    if (nextPos == pos) {
      ++nextPos;
      continue;
    }

    if (!stopCondition(cstr.positionToValueDim[nextPos].first)) {
      cstr.projectOut(nextPos);
      // The column was projected out so another column is now at that position.
      // Do not increase the counter.
    } else {
      ++nextPos;
    }
  }

  // Compute lower and upper bounds for `value`.
  SmallVector<AffineMap> lb(1), ub(1);
  cstr.cstr.getSliceBounds(pos, 1, b.getContext(), &lb, &ub,
                           /*getClosedUB=*/true);

  // Note: There are TODOs in the implementation of `getSliceBounds`. In such a
  // case, no lower/upper bound can be computed at the moment.
  // EQ, UB bounds: upper bound is needed.
  if ((type != presburger::IntegerPolyhedron::BoundType::LB) &&
      (ub.empty() || !ub[0] || ub[0].getNumResults() != 1))
    return failure();
  // EQ, LB bounds: lower bound is needed.
  if ((type != presburger::IntegerPolyhedron::BoundType::UB) &&
      (lb.empty() || !lb[0] || lb[0].getNumResults() != 1))
    return failure();

  // EQ bound: lower and upper bound must match.
  if (type == presburger::IntegerPolyhedron::BoundType::EQ && ub[0] != lb[0])
    return failure();

  AffineMap bound;
  if (type == presburger::IntegerPolyhedron::BoundType::EQ ||
      type == presburger::IntegerPolyhedron::BoundType::LB) {
    bound = lb[0];
  } else {
    // Computed UB is a closed bound. Turn into an open bound.
    bound = AffineMap::get(ub[0].getNumDims(), ub[0].getNumSymbols(),
                           ub[0].getResult(0) + 1);
  }

  // Gather all SSA values that are used in the computed bound.
  SmallVector<Value> operands;
  assert(cstr.cstr.getNumDimAndSymbolVars() == cstr.positionToValueDim.size() &&
         "inconsistent mapping state");
  for (int64_t i = 0; i < cstr.cstr.getNumDimAndSymbolVars(); ++i) {
    // Skip `value`.
    if (i == pos)
      continue;
    // Check if the position `i` is used in the generated bound. If so, it must
    // be included in the generated affine.apply op.
    bool used = false;
    if (i < cstr.cstr.getNumDimVars()) {
      if (bound.isFunctionOfDim(i))
        used = true;
    } else {
      if (bound.isFunctionOfSymbol(i - cstr.cstr.getNumDimVars()))
        used = true;
    }

    if (!used) {
      // Not used: Put an empty Value (will canonicalize away).
      operands.push_back(Value());
      continue;
    }

    ValueBoundsConstraintSet::ValueDim valueDim = cstr.positionToValueDim[i];
    Value value = valueDim.first;
    int64_t dim = valueDim.second;
    if (dim == ValueBoundsConstraintSet::kIndexValue) {
      // An index-type value is used: can be used directly in the affine.apply
      // op.
      assert(value.getType().isIndex() && "expected index type");
      operands.push_back(value);
      continue;
    }

    assert(value.getType().cast<ShapedType>().isDynamicDim(dim) &&
           "expected dynamic dim");
    if (value.getType().isa<RankedTensorType>()) {
      // A tensor dimension is used: generate a tensor.dim.
      operands.push_back(b.create<tensor::DimOp>(loc, value, dim));
    } else if (value.getType().isa<MemRefType>()) {
      // A memref dimension is used: generate a memref.dim.
      operands.push_back(b.create<memref::DimOp>(loc, value, dim));
    } else {
      llvm_unreachable("cannot generate DimOp for unsupported shaped type");
    }
  }

  mlir::canonicalizeMapAndOperands(&bound, &operands);
  // Check for special cases where no affine.apply op is needed.
  if (bound.isSingleConstant()) {
    // Bound is a constant: return an IntegerAttr.
    return static_cast<OpFoldResult>(
        b.getIndexAttr(bound.getSingleConstantResult()));
  }
  // No affine.apply op is needed if the bound is a single SSA value.
  if (auto expr = bound.getResult(0).dyn_cast<AffineDimExpr>())
    return static_cast<OpFoldResult>(operands[expr.getPosition()]);
  if (auto expr = bound.getResult(0).dyn_cast<AffineSymbolExpr>())
    return static_cast<OpFoldResult>(
        operands[expr.getPosition() + cstr.cstr.getNumDimVars() - 1]);
  // General case: build affine.apply op.
  return static_cast<OpFoldResult>(
      b.create<AffineApplyOp>(loc, bound, operands).getResult());
}
