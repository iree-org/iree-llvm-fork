// RUN: mlir-opt -rewrite-address-computation %s --split-input-file | FileCheck %s
// TODO: run with expand-strided-metadata + decompose affine

// TODO: check the lowering to affine map.
// The resulting address computation is:
// %offset * 16 * 16 + 0 * 16 + 8

// CHECK-LABEL: @test
// CHECK-SAME: (%[[BASE:.*]]: memref{{[^,]*}},
// CHECK-SAME: %[[DYN_OFFSET:.*]]: index)
// CHECK: %[[SUBVIEW:.*]] = memref.subview %[[BASE]][%[[DYN_OFFSET]], 0, 8] [1, 1, 1] [1, 1, 1] : memref<2x16x16xf32> to memref<1x1x1xf32, strided<[256, 16, 1], offset: ?>> 
// CHECK: %[[C0:.*]] = arith.constant 0 : index 
// CHECK: %[[LOADED_VAL:.*]] = memref.load %[[SUBVIEW]][%[[C0]], %[[C0]], %[[C0]]] : memref<1x1x1xf32, strided<[256, 16, 1], offset: ?>> 
// CHECK: return %[[LOADED_VAL]] : f32
func.func @test(%base : memref<2x16x16xf32>, %offset : index) -> f32 {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %loaded_val = memref.load %base[%offset, %c0, %c8] : memref<2x16x16xf32>
  return %loaded_val : f32
}

// -----

func.func @test_ldmatrix(%base : memref<4x32x32xf16, 3>, %offset0 : index, %offset1: index, %offset2: index) -> vector<4x2xf16> {
  %loaded_val = nvgpu.ldmatrix %base[%offset0, %offset1, %offset2] {numTiles = 4 : i32, transpose = false} : memref<4x32x32xf16, 3> -> vector<4x2xf16>
  return %loaded_val : vector<4x2xf16>
}
// -----

// Note: the scf.for are purposely flipped (dim2 -> dim0 instead of dim0 -> dim2) to
// make the ordering from the decompose of affine ops more obvious.
func.func @testWithLoop(%base : memref<?x?x?xf32, strided<[?,?,?], offset: ?>>) -> f32 {
  %sum_all = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %upper_bound0 = memref.dim %base, %c0 : memref<?x?x?xf32, strided<[?,?,?], offset: ?>>
  %upper_bound1 = memref.dim %base, %c1 : memref<?x?x?xf32, strided<[?,?,?], offset: ?>>
  %upper_bound2 = memref.dim %base, %c2 : memref<?x?x?xf32, strided<[?,?,?], offset: ?>>
  %sum_res2 = scf.for %iv2 = %c0 to %upper_bound2 step %c1 iter_args(%sum_iter2 = %sum_all) -> (f32) {
    %sum_res1 = scf.for %iv1 = %c0 to %upper_bound1 step %c1 iter_args(%sum_iter1 = %sum_iter2) -> (f32) {
      %sum_res0 = scf.for %iv0 = %c0 to %upper_bound0 step %c1 iter_args(%sum_iter0 = %sum_iter1) -> (f32) {
        %loaded_val = memref.load %base[%iv0, %iv1, %iv2] : memref<?x?x?xf32, strided<[?,?,?], offset: ?>>
        %res = arith.addf %loaded_val, %sum_iter2 : f32
        scf.yield %res : f32
      }
      scf.yield %sum_res0 : f32
    }
    scf.yield %sum_res1 : f32
  }
  return %sum_res2 : f32
}


