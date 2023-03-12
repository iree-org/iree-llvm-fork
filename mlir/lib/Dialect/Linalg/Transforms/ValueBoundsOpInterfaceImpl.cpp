//===- ValueBoundsOpInterfaceImpl.cpp - Impl. of ValueBoundsOpInterface ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/Transforms/ValueBoundsOpInterfaceImpl.h"

#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/ValueBoundsOpInterface.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

using namespace mlir;
using namespace mlir::linalg;
using presburger::IntegerPolyhedron;

namespace mlir {
namespace tensor {
namespace {

struct CastOpInterface
    : public ValueBoundsOpInterface::ExternalModel<CastOpInterface, CastOp> {
  void populateBoundsForShapedValueDim(Operation *op, Value value, int64_t dim,
                                       ValueBoundsConstraintSet &cstr) const {
    auto castOp = cast<CastOp>(op);
    assert(value == castOp.getResult() && "invalid value");

    if (castOp.getResult().getType().isa<RankedTensorType>() &&
        castOp.getSource().getType().isa<RankedTensorType>()) {
      cstr.addBound(IntegerPolyhedron::BoundType::EQ, value, dim,
                    cstr.getExpr(castOp.getSource(), dim));
    }
  }
};

struct DimOpInterface
    : public ValueBoundsOpInterface::ExternalModel<DimOpInterface, DimOp> {
  void populateBoundsForIndexValue(Operation *op, Value value,
                                   ValueBoundsConstraintSet &cstr) const {
    auto dimOp = cast<DimOp>(op);
    assert(value == dimOp.getResult() && "invalid value");

    auto constIndex = dimOp.getConstantIndex();
    if (!constIndex.has_value())
      return;
    cstr.addBound(IntegerPolyhedron::BoundType::EQ, value,
                  cstr.getExpr(dimOp.getSource(), *constIndex));
  }
};

struct EmptyOpInterface
    : public ValueBoundsOpInterface::ExternalModel<EmptyOpInterface, EmptyOp> {
  void populateBoundsForShapedValueDim(Operation *op, Value value, int64_t dim,
                                       ValueBoundsConstraintSet &cstr) const {
    auto emptyOp = cast<EmptyOp>(op);
    assert(value == emptyOp.getResult() && "invalid value");

    cstr.addBound(IntegerPolyhedron::BoundType::EQ, value, dim,
                  cstr.getExpr(emptyOp.getMixedSizes()[dim]));
  }
};

struct ExtractSliceOpInterface
    : public ValueBoundsOpInterface::ExternalModel<ExtractSliceOpInterface,
                                                   ExtractSliceOp> {
  void populateBoundsForShapedValueDim(Operation *op, Value value, int64_t dim,
                                       ValueBoundsConstraintSet &cstr) const {
    auto extractSliceOp = cast<ExtractSliceOp>(op);
    assert(value == extractSliceOp.getResult() && "invalid value");

    llvm::SmallBitVector dropped = extractSliceOp.getDroppedDims();
    int64_t ctr = -1;
    for (int64_t i = 0, e = extractSliceOp.getMixedSizes().size(); i < e; ++i) {
      // Skip over rank-reduced dimensions.
      if (!dropped.test(i))
        ++ctr;
      if (ctr == dim) {
        cstr.addBound(IntegerPolyhedron::BoundType::EQ, value, dim,
                      extractSliceOp.getMixedSizes()[i]);
        return;
      }
    }
    llvm_unreachable("could not find non-rank-reduced dim");
  }
};

struct PadOpInterface
    : public ValueBoundsOpInterface::ExternalModel<PadOpInterface, PadOp> {
  void populateBoundsForShapedValueDim(Operation *op, Value value, int64_t dim,
                                       ValueBoundsConstraintSet &cstr) const {
    auto padOp = cast<PadOp>(op);
    assert(value == padOp.getResult() && "invalid value");

    AffineExpr expr = cstr.getExpr(padOp.getSource(), dim) +
                      cstr.getExpr(padOp.getMixedLowPad()[dim]) +
                      cstr.getExpr(padOp.getMixedHighPad()[dim]);
    cstr.addBound(IntegerPolyhedron::BoundType::EQ, value, dim, expr);
  }
};

struct RankOpInterface
    : public ValueBoundsOpInterface::ExternalModel<RankOpInterface, RankOp> {
  void populateBoundsForIndexValue(Operation *op, Value value,
                                   ValueBoundsConstraintSet &cstr) const {
    auto rankOp = cast<RankOp>(op);
    assert(value == rankOp.getResult() && "invalid value");

    auto tensorType = rankOp.getTensor().getType().dyn_cast<RankedTensorType>();
    if (!tensorType)
      return;
    cstr.addBound(IntegerPolyhedron::BoundType::EQ, value,
                  cstr.getExpr(tensorType.getRank()));
  }
};

} // namespace
} // namespace tensor

namespace {
struct AffineApplyOpInterface
    : public ValueBoundsOpInterface::ExternalModel<AffineApplyOpInterface,
                                                   AffineApplyOp> {
  void populateBoundsForIndexValue(Operation *op, Value value,
                                   ValueBoundsConstraintSet &cstr) const {
    auto applyOp = cast<AffineApplyOp>(op);
    assert(value == applyOp.getResult() && "invalid value");
    assert(applyOp.getAffineMap().getNumResults() == 1 &&
           "expected single result");

    // Align affine map result with dims/symbols in the constraint set.
    AffineExpr expr = applyOp.getAffineMap().getResult(0);
    SmallVector<AffineExpr> dimReplacements = llvm::to_vector(llvm::map_range(
        applyOp.getDimOperands(), [&](Value v) { return cstr.getExpr(v); }));
    SmallVector<AffineExpr> symReplacements = llvm::to_vector(llvm::map_range(
        applyOp.getSymbolOperands(), [&](Value v) { return cstr.getExpr(v); }));
    AffineExpr bound =
        expr.replaceDimsAndSymbols(dimReplacements, symReplacements);
    cstr.addBound(IntegerPolyhedron::BoundType::EQ, value, bound);
  };
};

struct AffineMinOpInterface
    : public ValueBoundsOpInterface::ExternalModel<AffineMinOpInterface,
                                                   AffineMinOp> {
  void populateBoundsForIndexValue(Operation *op, Value value,
                                   ValueBoundsConstraintSet &cstr) const {
    auto minOp = cast<AffineMinOp>(op);
    assert(value == minOp.getResult() && "invalid value");

    // Align affine map results with dims/symbols in the constraint set.
    for (AffineExpr expr : minOp.getAffineMap().getResults()) {
      SmallVector<AffineExpr> dimReplacements = llvm::to_vector(llvm::map_range(
          minOp.getDimOperands(), [&](Value v) { return cstr.getExpr(v); }));
      SmallVector<AffineExpr> symReplacements = llvm::to_vector(llvm::map_range(
          minOp.getSymbolOperands(), [&](Value v) { return cstr.getExpr(v); }));
      AffineExpr bound =
          expr.replaceDimsAndSymbols(dimReplacements, symReplacements);
      AffineExpr openBound = bound + 1;
      cstr.addBound(IntegerPolyhedron::BoundType::UB, value, openBound);
    }
  };
};

struct AffineMaxOpInterface
    : public ValueBoundsOpInterface::ExternalModel<AffineMaxOpInterface,
                                                   AffineMaxOp> {
  void populateBoundsForIndexValue(Operation *op, Value value,
                                   ValueBoundsConstraintSet &cstr) const {
    auto maxOp = cast<AffineMaxOp>(op);
    assert(value == maxOp.getResult() && "invalid value");

    // Align affine map results with dims/symbols in the constraint set.
    for (AffineExpr expr : maxOp.getAffineMap().getResults()) {
      SmallVector<AffineExpr> dimReplacements = llvm::to_vector(llvm::map_range(
          maxOp.getDimOperands(), [&](Value v) { return cstr.getExpr(v); }));
      SmallVector<AffineExpr> symReplacements = llvm::to_vector(llvm::map_range(
          maxOp.getSymbolOperands(), [&](Value v) { return cstr.getExpr(v); }));
      AffineExpr bound =
          expr.replaceDimsAndSymbols(dimReplacements, symReplacements);
      cstr.addBound(IntegerPolyhedron::BoundType::LB, value, bound);
    }
  };
};
} // namespace

namespace memref {
namespace {

template <typename OpTy>
struct AllocOpInterface
    : public ValueBoundsOpInterface::ExternalModel<AllocOpInterface<OpTy>,
                                                   OpTy> {
  void populateBoundsForShapedValueDim(Operation *op, Value value, int64_t dim,
                                       ValueBoundsConstraintSet &cstr) const {
    auto allocOp = cast<OpTy>(op);
    assert(value == allocOp.getResult() && "invalid value");

    cstr.addBound(IntegerPolyhedron::BoundType::EQ, value, dim,
                  cstr.getExpr(allocOp.getMixedSizes()[dim]));
  }
};

struct CastOpInterface
    : public ValueBoundsOpInterface::ExternalModel<CastOpInterface, CastOp> {
  void populateBoundsForShapedValueDim(Operation *op, Value value, int64_t dim,
                                       ValueBoundsConstraintSet &cstr) const {
    auto castOp = cast<CastOp>(op);
    assert(value == castOp.getResult() && "invalid value");

    if (castOp.getResult().getType().isa<MemRefType>() &&
        castOp.getSource().getType().isa<MemRefType>()) {
      cstr.addBound(IntegerPolyhedron::BoundType::EQ, value, dim,
                    cstr.getExpr(castOp.getSource(), dim));
    }
  }
};

struct DimOpInterface
    : public ValueBoundsOpInterface::ExternalModel<DimOpInterface, DimOp> {
  void populateBoundsForIndexValue(Operation *op, Value value,
                                   ValueBoundsConstraintSet &cstr) const {
    auto dimOp = cast<DimOp>(op);
    assert(value == dimOp.getResult() && "invalid value");

    auto constIndex = dimOp.getConstantIndex();
    if (!constIndex.has_value())
      return;
    cstr.addBound(IntegerPolyhedron::BoundType::EQ, value,
                  cstr.getExpr(dimOp.getSource(), *constIndex));
  }
};

struct GetGlobalOpInterface
    : public ValueBoundsOpInterface::ExternalModel<GetGlobalOpInterface,
                                                   GetGlobalOp> {
  void populateBoundsForShapedValueDim(Operation *op, Value value, int64_t dim,
                                       ValueBoundsConstraintSet &cstr) const {
    auto getGlobalOp = cast<GetGlobalOp>(op);
    assert(value == getGlobalOp.getResult() && "invalid value");

    auto type = getGlobalOp.getType();
    assert(!type.isDynamicDim(dim) && "expected static dim");
    cstr.addBound(IntegerPolyhedron::BoundType::EQ, value, dim,
                  cstr.getExpr(type.getDimSize(dim)));
  }
};

struct RankOpInterface
    : public ValueBoundsOpInterface::ExternalModel<RankOpInterface, RankOp> {
  void populateBoundsForIndexValue(Operation *op, Value value,
                                   ValueBoundsConstraintSet &cstr) const {
    auto rankOp = cast<RankOp>(op);
    assert(value == rankOp.getResult() && "invalid value");

    auto memrefType = rankOp.getMemref().getType().dyn_cast<MemRefType>();
    if (!memrefType)
      return;
    cstr.addBound(IntegerPolyhedron::BoundType::EQ, value,
                  cstr.getExpr(memrefType.getRank()));
  }
};

struct SubViewOpInterface
    : public ValueBoundsOpInterface::ExternalModel<SubViewOpInterface,
                                                   SubViewOp> {
  void populateBoundsForShapedValueDim(Operation *op, Value value, int64_t dim,
                                       ValueBoundsConstraintSet &cstr) const {
    auto subViewOp = cast<SubViewOp>(op);
    assert(value == subViewOp.getResult() && "invalid value");

    llvm::SmallBitVector dropped = subViewOp.getDroppedDims();
    int64_t ctr = -1;
    for (int64_t i = 0, e = subViewOp.getMixedSizes().size(); i < e; ++i) {
      // Skip over rank-reduced dimensions.
      if (!dropped.test(i))
        ++ctr;
      if (ctr == dim) {
        cstr.addBound(IntegerPolyhedron::BoundType::EQ, value, dim,
                      subViewOp.getMixedSizes()[i]);
        return;
      }
    }
    llvm_unreachable("could not find non-rank-reduced dim");
  }
};

} // namespace
} // namespace memref

namespace arith {
namespace {

struct AddIOpInterface
    : public ValueBoundsOpInterface::ExternalModel<AddIOpInterface, AddIOp> {
  void populateBoundsForIndexValue(Operation *op, Value value,
                                   ValueBoundsConstraintSet &cstr) const {
    auto addIOp = cast<AddIOp>(op);
    assert(value == addIOp.getResult() && "invalid value");

    cstr.addBound(IntegerPolyhedron::BoundType::EQ, value,
                  cstr.getExpr(addIOp.getLhs()) +
                      cstr.getExpr(addIOp.getRhs()));
  }
};

struct SubIOpInterface
    : public ValueBoundsOpInterface::ExternalModel<SubIOpInterface, SubIOp> {
  void populateBoundsForIndexValue(Operation *op, Value value,
                                   ValueBoundsConstraintSet &cstr) const {
    auto subIOp = cast<SubIOp>(op);
    assert(value == subIOp.getResult() && "invalid value");

    cstr.addBound(IntegerPolyhedron::BoundType::EQ, value,
                  cstr.getExpr(subIOp.getLhs()) -
                      cstr.getExpr(subIOp.getRhs()));
  }
};

struct MulIOpInterface
    : public ValueBoundsOpInterface::ExternalModel<MulIOpInterface, MulIOp> {
  void populateBoundsForIndexValue(Operation *op, Value value,
                                   ValueBoundsConstraintSet &cstr) const {
    auto mulIOp = cast<MulIOp>(op);
    assert(value == mulIOp.getResult() && "invalid value");

    cstr.addBound(IntegerPolyhedron::BoundType::EQ, value,
                  cstr.getExpr(mulIOp.getLhs()) *
                      cstr.getExpr(mulIOp.getRhs()));
  }
};

} // namespace
} // namespace arith

namespace scf {
namespace {

struct ForOpInterface
    : public ValueBoundsOpInterface::ExternalModel<ForOpInterface, ForOp> {
  void populateBoundsForIndexValue(Operation *op, Value value,
                                   ValueBoundsConstraintSet &cstr) const {
    auto forOp = cast<ForOp>(op);
    // Only IV is supported at the moment.
    if (value != forOp.getInductionVar())
      return;

    // TODO: Take into account step size.
    cstr.addBound(IntegerPolyhedron::BoundType::LB, value,
                  cstr.getExpr(forOp.getLowerBound()));
    cstr.addBound(IntegerPolyhedron::BoundType::UB, value,
                  cstr.getExpr(forOp.getUpperBound()));
  }

  void populateBoundsForShapedValueDim(Operation *op, Value value, int64_t dim,
                                       ValueBoundsConstraintSet &cstr) const {
    // iter_arg / return value not supported.
    return;
  }
};

} // namespace
} // namespace scf

namespace linalg {
namespace {

/// Helper structure that iterates over all LinalgOps in `OpTys` and registers
/// the `BufferizableOpInterface` with each of them.
template <typename... Ops>
struct LinalgValueBoundsOpInterfaceHelper {
  static void registerOpInterface(MLIRContext *ctx) {
    (Ops::template attachInterface<DstValueBoundsOpInterfaceExternalModel<Ops>>(*ctx), ...);
  }
};

} // namespace
} // namespace linalg
} // namespace mlir

void mlir::linalg::registerValueBoundsOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, tensor::TensorDialect *dialect) {
    tensor::CastOp::attachInterface<tensor::CastOpInterface>(*ctx);
    tensor::DimOp::attachInterface<tensor::DimOpInterface>(*ctx);
    tensor::EmptyOp::attachInterface<tensor::EmptyOpInterface>(*ctx);
    tensor::ExtractSliceOp::attachInterface<tensor::ExtractSliceOpInterface>(
        *ctx);
    tensor::InsertOp::attachInterface<
        DstValueBoundsOpInterfaceExternalModel<tensor::InsertOp>>(*ctx);
    tensor::InsertSliceOp::attachInterface<
        DstValueBoundsOpInterfaceExternalModel<tensor::InsertSliceOp>>(*ctx);
    tensor::PadOp::attachInterface<tensor::PadOpInterface>(*ctx);
    tensor::RankOp::attachInterface<tensor::RankOpInterface>(*ctx);
  });

  registry.addExtension(+[](MLIRContext *ctx, AffineDialect *dialect) {
    AffineApplyOp::attachInterface<AffineApplyOpInterface>(*ctx);
    AffineMaxOp::attachInterface<AffineMaxOpInterface>(*ctx);
    AffineMinOp::attachInterface<AffineMinOpInterface>(*ctx);
  });

  registry.addExtension(+[](MLIRContext *ctx, memref::MemRefDialect *dialect) {
    memref::AllocOp::attachInterface<memref::AllocOpInterface<memref::AllocOp>>(
        *ctx);
    memref::AllocaOp::attachInterface<
        memref::AllocOpInterface<memref::AllocaOp>>(*ctx);
    memref::CastOp::attachInterface<memref::CastOpInterface>(*ctx);
    memref::DimOp::attachInterface<memref::DimOpInterface>(*ctx);
    memref::GetGlobalOp::attachInterface<memref::GetGlobalOpInterface>(*ctx);
    memref::RankOp::attachInterface<memref::RankOpInterface>(*ctx);
    memref::SubViewOp::attachInterface<memref::SubViewOpInterface>(*ctx);
  });

  registry.addExtension(+[](MLIRContext *ctx, arith::ArithDialect *dialect) {
    arith::AddIOp::attachInterface<arith::AddIOpInterface>(*ctx);
    arith::SubIOp::attachInterface<arith::SubIOpInterface>(*ctx);
    arith::MulIOp::attachInterface<arith::MulIOpInterface>(*ctx);
  });

  registry.addExtension(+[](MLIRContext *ctx, scf::SCFDialect *dialect) {
    scf::ForOp::attachInterface<scf::ForOpInterface>(*ctx);
  });

  registry.addExtension(+[](MLIRContext *ctx, linalg::LinalgDialect *dialect) {
    // Register all Linalg structured ops.
    LinalgValueBoundsOpInterfaceHelper<
#define GET_OP_LIST
#include "mlir/Dialect/Linalg/IR/LinalgStructuredOps.cpp.inc"
        >::registerOpInterface(ctx);
  });
}
