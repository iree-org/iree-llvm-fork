//===- Transforms.h - Transforms Entrypoints --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines a set of transforms specific for the AffineOps
// dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_AFFINE_TRANSFORMS_TRANSFORMS_H
#define MLIR_DIALECT_AFFINE_TRANSFORMS_TRANSFORMS_H

#include "mlir/Support/LogicalResult.h"

namespace mlir {
class AffineApplyOp;
class Location;
class OpBuilder;
class OpFoldResult;
class RewritePatternSet;
class RewriterBase;
class Value;

namespace presburger {
enum class BoundType;
} // namespace presburger

/// Populate patterns that expand affine index operations into more fundamental
/// operations (not necessarily restricted to Affine dialect).
void populateAffineExpandIndexOpsPatterns(RewritePatternSet &patterns);

/// Helper function to rewrite `op`'s affine map and reorder its operands such
/// that they are in increasing order of hoistability (i.e. the least hoistable)
/// operands come first in the operand list.
void reorderOperandsByHoistability(RewriterBase &rewriter, AffineApplyOp op);

/// Split an "affine.apply" operation into smaller ops.
/// This reassociates a large AffineApplyOp into an ordered list of smaller
/// AffineApplyOps. This can be used right before lowering affine ops to arith
/// to exhibit more opportunities for CSE and LICM.
/// Return the sink AffineApplyOp on success or failure if `op` does not
/// decompose into smaller AffineApplyOps.
/// Note that this can be undone by canonicalization which tries to
/// maximally compose chains of AffineApplyOps.
FailureOr<AffineApplyOp> decompose(RewriterBase &rewriter, AffineApplyOp op);

/// Reify a bound for the given index-typed value or shape dimension size in
/// terms of the owning op's operands. `dim` must be `nullopt` if and only if
/// `value` is index-typed. LB and EQ bounds are closed, UB bounds are open.
FailureOr<OpFoldResult> reifyValueBound(OpBuilder &b, Location loc,
                                        presburger::BoundType type, Value value,
                                        std::optional<int64_t> dim);

/// Reify a bound for the given index-typed value or shape dimension size in
/// terms of SSA values for which `stopCondition` is met. `dim` must be
/// `nullopt` if and only if `value` is index-typed. LB and EQ bounds are
/// closed, UB bounds are open.
FailureOr<OpFoldResult> reifyValueBound(
    OpBuilder &b, Location loc, presburger::BoundType type, Value value,
    std::optional<int64_t> dim,
    function_ref<bool(Value, std::optional<int64_t>)> stopCondition);

} // namespace mlir

#endif // MLIR_DIALECT_AFFINE_TRANSFORMS_TRANSFORMS_H
