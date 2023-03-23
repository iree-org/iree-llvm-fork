//===- VectorTransformOps.td - Vector transform ops --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef VECTOR_TRANSFORM_OPS
#define VECTOR_TRANSFORM_OPS

include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/IR/TransformInterfaces.td"
include "mlir/Dialect/PDL/IR/PDLTypes.td"
include "mlir/Dialect/Vector/Transforms/VectorTransformsBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"


// TODO: not isolated from above and better targetability of the op.
// TODO: not a functional-style transform.
def ApplyRankReducingSubviewPatternsOp : 
  Op<Transform_Dialect, "vector.apply_rank_reducing_subview_patterns",
    [FunctionalStyleTransformOpTrait, 
     MemoryEffectsOpInterface,
     TransformOpInterface, 
     TransformEachOpTrait]> {
  let description = [{
    Apply opt-in vector transfer permutation patterns that include:
      - TransferReadDropUnitDimsPattern
      - TransferWriteDropUnitDimsPattern
    
    These patterns have the effect of rewriting a vector.transfer with unit 
    dimensions into a rank-reduced version thanks to subview operations.
    This is complemented by shape_cast folding patterns.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target);
  let results = (outs TransformHandleTypeInterface:$results);

  let assemblyFormat = [{
    $target
    attr-dict
    `:` functional-type($target, results)
  }];
  
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

// TODO: not isolated from above and better targetability of the op.
// TODO: not a functional-style transform.
def ApplyTransferPermutationPatternsOp : 
  Op<Transform_Dialect, "vector.apply_transfer_permutation_patterns",
    [FunctionalStyleTransformOpTrait, 
     MemoryEffectsOpInterface,
     TransformOpInterface, 
     TransformEachOpTrait]> {
  let description = [{
    Apply opt-in vector transfer permutation patterns that include:
      - TransferReadPermutationLowering
      - TransferWritePermutationLowering
      - TransferOpReduceRank
      - TransferWriteNonPermutationLowering
    
    These patterns have the effect of rewriting a vector.transfer with an 
    aritrary permutation_map to a vector.transfer with a permutation_map that is
    a minor identity followed by a vector.transpose.

    In other words, this makes the vector.transfer contiguous on the most minor
    dimensions and materializes the permutation_map as a vector.transpose.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target);
  let results = (outs TransformHandleTypeInterface:$results);

  let assemblyFormat = [{
    $target
    attr-dict
    `:` functional-type($target, results)
  }];
  
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

// TODO: not isolated from above and better targetability of the op.
// TODO: not a functional-style transform.
// TODO: evolve lowering_strategy to proper enums.
def LowerBroadcastOp : Op<Transform_Dialect, "vector.lower_broadcast",
    [FunctionalStyleTransformOpTrait, 
     MemoryEffectsOpInterface,
     TransformOpInterface, 
     TransformEachOpTrait]> {
  let description = [{
    Indicates that the vector outerproduct operations nested under the isolated
    from above op `target` should be lowered to finer-grained vector primitives.

    This is usally a late step that is run after bufferization as part of the
    process of lowering to e.g. LLVM or NVVM.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target
  );
  let results = (outs TransformHandleTypeInterface:$results);

  let assemblyFormat = [{
    $target
    attr-dict
    `:` functional-type($target, results)
  }];
  
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

// TODO: not isolated from above and better targetability of the op.
// TODO: not a functional-style transform.
// TODO: evolve lowering_strategy to proper enums.
def LowerContractionOp : Op<Transform_Dialect, "vector.lower_contraction",
    [FunctionalStyleTransformOpTrait, 
     MemoryEffectsOpInterface,
     TransformOpInterface, 
     TransformEachOpTrait]> {
  let description = [{
    Indicates that the vector contraction-like operations nested under the 
    isolated from above op `target` should be lowered to finer-grained vector
    primitives.

    This is usally a late step that is run after bufferization as part of the
    process of lowering to e.g. LLVM or NVVM.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target,
     DefaultValuedAttr<VectorContractLoweringAttr,
       "vector::VectorContractLowering::OuterProduct">:$lowering_strategy
  );
  let results = (outs TransformHandleTypeInterface:$results);

  let assemblyFormat = [{
    $target
    (`lowering_strategy` `=` $lowering_strategy^)?
    attr-dict
    `:` functional-type($target, results)
  }];
  
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

// TODO: not isolated from above and better targetability of the op.
// TODO: not a functional-style transform.
// TODO: evolve lowering_strategy to proper enums.
def LowerMaskOp : Op<Transform_Dialect, "vector.lower_mask",
    [FunctionalStyleTransformOpTrait, 
     MemoryEffectsOpInterface,
     TransformOpInterface, 
     TransformEachOpTrait]> {
  let description = [{
    Indicates that the vector mask operations nested under the isolated from
    above op `target` should be lowered to finer-grained vector primitives.

    This is usally a late step that is run after bufferization as part of the
    process of lowering to e.g. LLVM or NVVM.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target
  );
  let results = (outs TransformHandleTypeInterface:$results);

  let assemblyFormat = [{
    $target
    attr-dict
    `:` functional-type($target, results)
  }];
  
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

// TODO: not isolated from above and better targetability of the op.
// TODO: not a functional-style transform.
// TODO: evolve lowering_strategy to proper enums.
def LowerMultiReductionOp : Op<Transform_Dialect, "vector.lower_multi_reduction",
    [FunctionalStyleTransformOpTrait, 
     MemoryEffectsOpInterface,
     TransformOpInterface, 
     TransformEachOpTrait]> {
  let description = [{
    Indicates that the vector multi_reduction-like operations nested under the 
    isolated from above op `target` should be lowered to finer-grained vector
    primitives.

    This is usally a late step that is run after bufferization as part of the
    process of lowering to e.g. LLVM or NVVM.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target,
     DefaultValuedAttr<VectorMultiReductionLoweringAttr,
       "vector::VectorMultiReductionLowering::InnerParallel">:
         $lowering_strategy
  );
  let results = (outs TransformHandleTypeInterface:$results);

  let assemblyFormat = [{
    $target
    (`lowering_strategy` `=` $lowering_strategy^)?
    attr-dict
    `:` functional-type($target, results)
  }];
  
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

// TODO: not isolated from above and better targetability of the op.
// TODO: not a functional-style transform.
// TODO: evolve lowering_strategy to proper enums.
def LowerOuterProductOp : Op<Transform_Dialect, "vector.lower_outerproduct",
    [FunctionalStyleTransformOpTrait, 
     MemoryEffectsOpInterface,
     TransformOpInterface, 
     TransformEachOpTrait]> {
  let description = [{
    Indicates that the vector outerproduct operations nested under the isolated
    from above op `target` should be lowered to finer-grained vector primitives.

    This is usally a late step that is run after bufferization as part of the
    process of lowering to e.g. LLVM or NVVM.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target
  );
  let results = (outs TransformHandleTypeInterface:$results);

  let assemblyFormat = [{
    $target
    attr-dict
    `:` functional-type($target, results)
  }];
  
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

// TODO: not isolated from above and better targetability of the op.
// TODO: not a functional-style transform.
def LowerShapeCastOp : Op<Transform_Dialect, "vector.lower_shape_cast",
    [FunctionalStyleTransformOpTrait, 
     MemoryEffectsOpInterface,
     TransformOpInterface, 
     TransformEachOpTrait]> {
  let description = [{
    Indicates that the vector shape_cast operations nested under the 
    isolated from above op `target` should be lowered to finer-grained vector
    primitives.

    This is usally a late step that is run after bufferization as part of the
    process of lowering to e.g. LLVM or NVVM.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target);
  let results = (outs TransformHandleTypeInterface:$results);

  let assemblyFormat = [{
    $target
    attr-dict
    `:` functional-type($target, results)
  }];
  
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

// TODO: not isolated from above and better targetability of the op.
// TODO: not a functional-style transform.
def LowerTransferOp : Op<Transform_Dialect, "vector.lower_transfer",
    [FunctionalStyleTransformOpTrait, 
     MemoryEffectsOpInterface,
     TransformOpInterface, 
     TransformEachOpTrait]> {
  let description = [{
    Indicates that the vector transfer operations nested under the 
    isolated from above op `target` should be lowered to finer-grained vector
    primitives.

    This is usally a late step that is run after bufferization as part of the
    process of lowering to e.g. LLVM or NVVM.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target,
     DefaultValuedAttr<I64Attr, "1">:$max_transfer_rank
  );
  let results = (outs TransformHandleTypeInterface:$results);

  let assemblyFormat = [{
    $target
    (`max_transfer_rank` `=` $max_transfer_rank^)?
    attr-dict
    `:` functional-type($target, results)
  }];
  
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

// TODO: not isolated from above and better targetability of the op.
// TODO: not a functional-style transform.
// TODO: evolve lowering_strategy to proper enums.
def LowerTransposeOp : Op<Transform_Dialect, "vector.lower_transpose",
    [FunctionalStyleTransformOpTrait, 
     MemoryEffectsOpInterface,
     TransformOpInterface, 
     TransformEachOpTrait]> {
  let description = [{
    Indicates that the vector transpose-like operations nested under the 
    isolated from above op `target` should be lowered to finer-grained vector
    primitives.

    This is usally a late step that is run after bufferization as part of the
    process of lowering to e.g. LLVM or NVVM.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target,
     DefaultValuedAttr<VectorTransposeLoweringAttr,
       "vector::VectorTransposeLowering::EltWise">:$lowering_strategy,
     DefaultValuedAttr<BoolAttr, "false">:$avx2_lowering_strategy
  );
  let results = (outs TransformHandleTypeInterface:$results);

  let assemblyFormat = [{
    $target
    oilist (
      `lowering_strategy` `=` $lowering_strategy
      | `avx2_lowering_strategy` `=` $avx2_lowering_strategy
    )
    attr-dict
    `:` functional-type($target, results)
  }];
  
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

// TODO: not isolated from above and better targetability of the op.
// TODO: not a functional-style transform.
// TODO: evolve split_transfer_strategy to proper enums.
def SplitTransferFullPartialOp : Op<Transform_Dialect, "vector.split_transfer_full_partial",
    [FunctionalStyleTransformOpTrait, 
     MemoryEffectsOpInterface,
     TransformOpInterface, 
     TransformEachOpTrait]> {
  let description = [{
    Indicates that the vector transfer operations nested under the 
    isolated from above op `target` should be split to full and partial parts.

    This is usally a late step that is run after bufferization as part of the
    process of lowering to e.g. LLVM or NVVM.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target,
     DefaultValuedAttr<VectorTransferSplitAttr,
       "vector::VectorTransferSplit::LinalgCopy">:$split_transfer_strategy
  );
  let results = (outs TransformHandleTypeInterface:$results);

  let assemblyFormat = [{
    $target
    (`split_transfer_strategy` `=` $split_transfer_strategy^)?
    attr-dict
    `:` functional-type($target, results)
  }];
  
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

// TODO: not isolated from above and better targetability of the op.
// TODO: not a functional-style transform.
def TransferToScfOp : Op<Transform_Dialect, "vector.transfer_to_scf",
    [FunctionalStyleTransformOpTrait, 
     MemoryEffectsOpInterface,
     TransformOpInterface, 
     TransformEachOpTrait]> {
  let description = [{
    Indicates that the vector transfer operations nested under the 
    isolated from above op `target` should be rewritten with scf.for loops over
    finer-grained vector primitives.

    This is usally a late step that is run after bufferization as part of the
    process of lowering to e.g. LLVM or NVVM.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target,
     DefaultValuedAttr<I64Attr, "1">:$max_transfer_rank,
     DefaultValuedAttr<BoolAttr, "false">:$full_unroll
  );
  let results = (outs TransformHandleTypeInterface:$results);

  let assemblyFormat = [{
    $target
    oilist (
        `max_transfer_rank` `=` $max_transfer_rank
      | `full_unroll` `=` $full_unroll
    )
    attr-dict
    `:` functional-type($target, results)
  }];
  
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

#endif // VECTOR_TRANSFORM_OPS
