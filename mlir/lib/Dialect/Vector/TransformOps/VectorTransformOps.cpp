//===- VectorTransformOps.cpp - Implementation of Vector transform ops ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Vector/TransformOps/VectorTransformOps.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDL/IR/PDLTypes.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Dialect/X86Vector/Transforms.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::vector;
using namespace mlir::transform;

//===----------------------------------------------------------------------===//
// ApplyRankReducingSubviewPatternsOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::ApplyRankReducingSubviewPatternsOp::applyToOne(
    ::mlir::Operation *target,
    ::mlir::transform::ApplyToEachResultList &results,
    ::mlir::transform::TransformState &state) {
  // This check can't be part of the verifier because payload IR is
  // independent from transform IR and may not even exist.
  if (!target->hasTrait<OpTrait::IsIsolatedFromAbove>()) {
    return mlir::emitDefiniteFailure(target,
                                     "applies only to isolated-from-above "
                                     "targets because it needs to apply "
                                     "patterns greedily");
  }

  RewritePatternSet patterns(getContext());
  vector::populateVectorTransferDropUnitDimsPatterns(patterns);

  if (failed(applyPatternsAndFoldGreedily(target, std::move(patterns))))
    return DiagnosedSilenceableFailure::definiteFailure();

  results.push_back(target);
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// ApplyTransferPermutationPatternsOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::ApplyTransferPermutationPatternsOp::applyToOne(
    ::mlir::Operation *target,
    ::mlir::transform::ApplyToEachResultList &results,
    ::mlir::transform::TransformState &state) {
  // This check can't be part of the verifier because payload IR is
  // independent from transform IR and may not even exist.
  if (!target->hasTrait<OpTrait::IsIsolatedFromAbove>()) {
    return mlir::emitDefiniteFailure(target,
                                     "applies only to isolated-from-above "
                                     "targets because it needs to apply "
                                     "patterns greedily");
  }

  RewritePatternSet patterns(getContext());
  vector::populateVectorTransferPermutationMapLoweringPatterns(patterns);

  if (failed(applyPatternsAndFoldGreedily(target, std::move(patterns))))
    return DiagnosedSilenceableFailure::definiteFailure();

  results.push_back(target);
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// LowerBroadcastOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure transform::LowerBroadcastOp::applyToOne(
    ::mlir::Operation *target,
    ::mlir::transform::ApplyToEachResultList &results,
    ::mlir::transform::TransformState &state) {
  // This check can't be part of the verifier because payload IR is
  // independent from transform IR and may not even exist.
  if (!target->hasTrait<OpTrait::IsIsolatedFromAbove>()) {
    return mlir::emitDefiniteFailure(target,
                                     "applies only to isolated-from-above "
                                     "targets because it needs to apply "
                                     "patterns greedily");
  }

  RewritePatternSet patterns(getContext());
  populateVectorBroadcastLoweringPatterns(patterns);

  if (failed(applyPatternsAndFoldGreedily(target, std::move(patterns))))
    return DiagnosedSilenceableFailure::definiteFailure();

  results.push_back(target);
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// LowerContractionOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure transform::LowerContractionOp::applyToOne(
    ::mlir::Operation *target,
    ::mlir::transform::ApplyToEachResultList &results,
    ::mlir::transform::TransformState &state) {
  // This check can't be part of the verifier because payload IR is
  // independent from transform IR and may not even exist.
  if (!target->hasTrait<OpTrait::IsIsolatedFromAbove>()) {
    return mlir::emitDefiniteFailure(target,
                                     "applies only to isolated-from-above "
                                     "targets because it needs to apply "
                                     "patterns greedily");
  }

  RewritePatternSet patterns(getContext());
  vector::VectorTransformsOptions vectorTransformOptions;
  vectorTransformOptions.setVectorTransformsOptions(getLoweringStrategy());
  populateVectorContractLoweringPatterns(patterns, vectorTransformOptions,
                                         /*benefit=*/1,
                                         /*disableOuterProductLowering=*/true);

  if (failed(applyPatternsAndFoldGreedily(target, std::move(patterns))))
    return DiagnosedSilenceableFailure::definiteFailure();

  results.push_back(target);
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// LowerMaskOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure transform::LowerMaskOp::applyToOne(
    ::mlir::Operation *target,
    ::mlir::transform::ApplyToEachResultList &results,
    ::mlir::transform::TransformState &state) {
  // This check can't be part of the verifier because payload IR is
  // independent from transform IR and may not even exist.
  if (!target->hasTrait<OpTrait::IsIsolatedFromAbove>()) {
    return mlir::emitDefiniteFailure(target,
                                     "applies only to isolated-from-above "
                                     "targets because it needs to apply "
                                     "patterns greedily");
  }

  RewritePatternSet patterns(getContext());
  populateVectorMaskOpLoweringPatterns(patterns);
  populateVectorMaskLoweringPatternsForSideEffectingOps(patterns);

  if (failed(applyPatternsAndFoldGreedily(target, std::move(patterns))))
    return DiagnosedSilenceableFailure::definiteFailure();

  results.push_back(target);
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// LowerMultiReductionOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure transform::LowerMultiReductionOp::applyToOne(
    ::mlir::Operation *target,
    ::mlir::transform::ApplyToEachResultList &results,
    ::mlir::transform::TransformState &state) {
  // This check can't be part of the verifier because payload IR is
  // independent from transform IR and may not even exist.
  if (!target->hasTrait<OpTrait::IsIsolatedFromAbove>()) {
    return mlir::emitDefiniteFailure(target,
                                     "applies only to isolated-from-above "
                                     "targets because it needs to apply "
                                     "patterns greedily");
  }

  RewritePatternSet patterns(getContext());
  vector::VectorTransformsOptions vectorTransformOptions;
  vectorTransformOptions.setVectorMultiReductionLowering(getLoweringStrategy());
  vector::populateVectorMultiReductionLoweringPatterns(
      patterns, vectorTransformOptions.vectorMultiReductionLowering);

  if (failed(applyPatternsAndFoldGreedily(target, std::move(patterns))))
    return DiagnosedSilenceableFailure::definiteFailure();

  results.push_back(target);
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// LowerOuterProductOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure transform::LowerOuterProductOp::applyToOne(
    ::mlir::Operation *target,
    ::mlir::transform::ApplyToEachResultList &results,
    ::mlir::transform::TransformState &state) {
  // This check can't be part of the verifier because payload IR is
  // independent from transform IR and may not even exist.
  if (!target->hasTrait<OpTrait::IsIsolatedFromAbove>()) {
    return mlir::emitDefiniteFailure(target,
                                     "applies only to isolated-from-above "
                                     "targets because it needs to apply "
                                     "patterns greedily");
  }

  RewritePatternSet patterns(getContext());
  populateVectorOuterProductLoweringPatterns(patterns);

  if (failed(applyPatternsAndFoldGreedily(target, std::move(patterns))))
    return DiagnosedSilenceableFailure::definiteFailure();

  results.push_back(target);
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// LowerShapeCastOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure transform::LowerShapeCastOp::applyToOne(
    ::mlir::Operation *target,
    ::mlir::transform::ApplyToEachResultList &results,
    ::mlir::transform::TransformState &state) {
  // This check can't be part of the verifier because payload IR is
  // independent from transform IR and may not even exist.
  if (!target->hasTrait<OpTrait::IsIsolatedFromAbove>()) {
    return mlir::emitDefiniteFailure(target,
                                     "applies only to isolated-from-above "
                                     "targets because it needs to apply "
                                     "patterns greedily");
  }

  RewritePatternSet patterns(getContext());
  vector::populateVectorShapeCastLoweringPatterns(patterns);

  if (failed(applyPatternsAndFoldGreedily(target, std::move(patterns))))
    return DiagnosedSilenceableFailure::definiteFailure();

  results.push_back(target);
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// LowerTransferOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure transform::LowerTransferOp::applyToOne(
    ::mlir::Operation *target,
    ::mlir::transform::ApplyToEachResultList &results,
    ::mlir::transform::TransformState &state) {
  // This check can't be part of the verifier because payload IR is
  // independent from transform IR and may not even exist.
  if (!target->hasTrait<OpTrait::IsIsolatedFromAbove>()) {
    return mlir::emitDefiniteFailure(target,
                                     "applies only to isolated-from-above "
                                     "targets because it needs to apply "
                                     "patterns greedily");
  }

  RewritePatternSet patterns(getContext());
  vector::populateVectorTransferLoweringPatterns(patterns,
                                                 getMaxTransferRank());

  if (failed(applyPatternsAndFoldGreedily(target, std::move(patterns))))
    return DiagnosedSilenceableFailure::definiteFailure();

  results.push_back(target);
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// LowerTransposeOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure transform::LowerTransposeOp::applyToOne(
    ::mlir::Operation *target,
    ::mlir::transform::ApplyToEachResultList &results,
    ::mlir::transform::TransformState &state) {
  // This check can't be part of the verifier because payload IR is
  // independent from transform IR and may not even exist.
  if (!target->hasTrait<OpTrait::IsIsolatedFromAbove>()) {
    return mlir::emitDefiniteFailure(target,
                                     "applies only to isolated-from-above "
                                     "targets because it needs to apply "
                                     "patterns greedily");
  }

  RewritePatternSet patterns(getContext());
  vector::populateVectorTransposeLoweringPatterns(
      patterns, vector::VectorTransformsOptions().setVectorTransposeLowering(
                    getLoweringStrategy()));
  if (getAvx2LoweringStrategy()) {
    auto avx2LoweringOptions =
        x86vector::avx2::LoweringOptions().setTransposeOptions(
            x86vector::avx2::TransposeLoweringOptions()
                .lower4x8xf32(true)
                .lower8x8xf32(true));
    x86vector::avx2::populateSpecializedTransposeLoweringPatterns(
        patterns, avx2LoweringOptions, /*benefit=*/10);
  }

  if (failed(applyPatternsAndFoldGreedily(target, std::move(patterns))))
    return DiagnosedSilenceableFailure::definiteFailure();

  results.push_back(target);
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// SplitTransferFullPartialOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure transform::SplitTransferFullPartialOp::applyToOne(
    ::mlir::Operation *target,
    ::mlir::transform::ApplyToEachResultList &results,
    ::mlir::transform::TransformState &state) {
  // This check can't be part of the verifier because payload IR is
  // independent from transform IR and may not even exist.
  if (!target->hasTrait<OpTrait::IsIsolatedFromAbove>()) {
    return mlir::emitDefiniteFailure(target,
                                     "applies only to isolated-from-above "
                                     "targets because it needs to apply "
                                     "patterns greedily");
  }

  RewritePatternSet patterns(getContext());
  vector::VectorTransformsOptions vectorTransformOptions;
  vectorTransformOptions.setVectorTransferSplit(getSplitTransferStrategy());
  populateVectorTransferFullPartialPatterns(patterns, vectorTransformOptions);

  if (failed(applyPatternsAndFoldGreedily(target, std::move(patterns))))
    return DiagnosedSilenceableFailure::definiteFailure();

  results.push_back(target);
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// TransferToScfOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure transform::TransferToScfOp::applyToOne(
    ::mlir::Operation *target,
    ::mlir::transform::ApplyToEachResultList &results,
    ::mlir::transform::TransformState &state) {
  // This check can't be part of the verifier because payload IR is
  // independent from transform IR and may not even exist.
  if (!target->hasTrait<OpTrait::IsIsolatedFromAbove>()) {
    return mlir::emitDefiniteFailure(target,
                                     "applies only to isolated-from-above "
                                     "targets because it needs to apply "
                                     "patterns greedily");
  }

  RewritePatternSet patterns(getContext());
  VectorTransferToSCFOptions vectorTransferToSCFOptions =
      VectorTransferToSCFOptions()
          .enableFullUnroll(getFullUnroll())
          .setTargetRank(getMaxTransferRank());
  populateVectorToSCFConversionPatterns(patterns, vectorTransferToSCFOptions);

  if (failed(applyPatternsAndFoldGreedily(target, std::move(patterns))))
    return DiagnosedSilenceableFailure::definiteFailure();

  results.push_back(target);
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// Transform op registration
//===----------------------------------------------------------------------===//

namespace {
/// Registers new ops and declares PDL as dependent dialect since the additional
/// ops are using PDL types for operands and results.
class VectorTransformDialectExtension
    : public transform::TransformDialectExtension<
          VectorTransformDialectExtension> {
public:
  VectorTransformDialectExtension() {
    declareDependentDialect<pdl::PDLDialect>();
    declareGeneratedDialect<vector::VectorDialect>();
    declareGeneratedDialect<LLVM::LLVMDialect>();
    registerTransformOps<
#define GET_OP_LIST
#include "mlir/Dialect/Vector/TransformOps/VectorTransformOps.cpp.inc"
        >();
  }
};
} // namespace

#define GET_OP_CLASSES
#include "mlir/Dialect/Vector/TransformOps/VectorTransformOps.cpp.inc"

void mlir::vector::registerTransformDialectExtension(
    DialectRegistry &registry) {
  registry.addExtensions<VectorTransformDialectExtension>();
}
