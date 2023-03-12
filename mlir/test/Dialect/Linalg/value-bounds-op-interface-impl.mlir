// RUN: mlir-opt %s -test-linalg-transform-patterns=test-refiy-shape-dims \
// RUN:     -verify-diagnostics -split-input-file -allow-unregistered-dialect | FileCheck %s

func.func @unknown_op() -> index {
  %0 = "test.foo"() : () -> (tensor<?x?xf32>)
  // expected-error @below{{could not reify bound}}
  %1 = "test.reify_bound"(%0) {dim = 0} : (tensor<?x?xf32>) -> (index)
  return %1 : index
}

// -----

// CHECK-LABEL: func @cast(
//       CHECK:   %[[c10:.*]] = arith.constant 10 : index
//       CHECK:   return %[[c10]]
func.func @cast(%t: tensor<10xf32>) -> index {
  %0 = tensor.cast %t : tensor<10xf32> to tensor<?xf32>
  %1 = "test.reify_bound"(%0) {dim = 0} : (tensor<?xf32>) -> (index)
  return %1 : index
}

// -----

func.func @cast_unranked(%t: tensor<*xf32>) -> index {
  %0 = tensor.cast %t : tensor<*xf32> to tensor<?xf32>
  // expected-error @below{{could not reify bound}}
  %1 = "test.reify_bound"(%0) {dim = 0} : (tensor<?xf32>) -> (index)
  return %1 : index
}

// -----

// CHECK-LABEL: func @dim(
//  CHECK-SAME:     %[[t:.*]]: tensor<?xf32>
//       CHECK:   %[[dim:.*]] = tensor.dim %[[t]]
//       CHECK:   %[[dim:.*]] = tensor.dim %[[t]]
//       CHECK:   return %[[dim]]
func.func @dim(%t: tensor<?xf32>) -> index {
  %c0 = arith.constant 0 : index
  %0 = tensor.dim %t, %c0 : tensor<?xf32>
  %1 = "test.reify_bound"(%0) : (index) -> (index)
  return %1 : index
}

// -----

// CHECK-LABEL: func @empty(
//  CHECK-SAME:     %[[sz:.*]]: index
//       CHECK:   %[[c6:.*]] = arith.constant 6 : index
//       CHECK:   return %[[c6]], %[[sz]]
func.func @empty(%sz: index) -> (index, index) {
  %0 = tensor.empty(%sz) : tensor<6x?xf32>
  %1 = "test.reify_bound"(%0) {dim = 0} : (tensor<6x?xf32>) -> (index)
  %2 = "test.reify_bound"(%0) {dim = 1} : (tensor<6x?xf32>) -> (index)
  return %1, %2 : index, index
}

// -----

// CHECK-LABEL: func @extract_slice_dynamic(
//  CHECK-SAME:     %[[t:.*]]: tensor<?xf32>, %[[sz:.*]]: index
//       CHECK:   return %[[sz]]
func.func @extract_slice_dynamic(%t: tensor<?xf32>, %sz: index) -> index {
  %0 = tensor.extract_slice %t[2][%sz][1] : tensor<?xf32> to tensor<?xf32>
  %1 = "test.reify_bound"(%0) {dim = 0} : (tensor<?xf32>) -> (index)
  return %1 : index
}

// -----

// CHECK-LABEL: func @extract_slice_static(
//  CHECK-SAME:     %[[t:.*]]: tensor<?xf32>
//       CHECK:   %[[c5:.*]] = arith.constant 5 : index
//       CHECK:   return %[[c5]]
func.func @extract_slice_static(%t: tensor<?xf32>) -> index {
  %0 = tensor.extract_slice %t[2][5][1] : tensor<?xf32> to tensor<5xf32>
  %1 = "test.reify_bound"(%0) {dim = 0} : (tensor<5xf32>) -> (index)
  return %1 : index
}

// -----

// CHECK-LABEL: func @extract_slice_rank_reduce(
//  CHECK-SAME:     %[[t:.*]]: tensor<?x?xf32>, %[[sz:.*]]: index
//       CHECK:   return %[[sz]]
func.func @extract_slice_rank_reduce(%t: tensor<?x?xf32>, %sz: index) -> index {
  %0 = tensor.extract_slice %t[0, 2][1, %sz][1, 1] : tensor<?x?xf32> to tensor<?xf32>
  %1 = "test.reify_bound"(%0) {dim = 0} : (tensor<?xf32>) -> (index)
  return %1 : index
}

// -----

// CHECK-LABEL: func @insert(
//  CHECK-SAME:     %[[t:.*]]: tensor<?xf32>
//       CHECK:   %[[c0:.*]] = arith.constant 0 : index
//       CHECK:   %[[dim:.*]] = tensor.dim %[[t]], %[[c0]]
//       CHECK:   return %[[dim]]
func.func @insert(%t: tensor<?xf32>, %f: f32, %pos: index) -> index {
  %0 = tensor.insert %f into %t[%pos] : tensor<?xf32>
  %1 = "test.reify_bound"(%0) {dim = 0} : (tensor<?xf32>) -> (index)
  return %1 : index
}

// -----

// CHECK: #[[$map:.*]] = affine_map<()[s0] -> (s0 + 12)>
// CHECK: #[[$map1:.*]] = affine_map<()[s0, s1] -> (s0 + s1 * 2)>
// CHECK-LABEL: func @pad(
//  CHECK-SAME:     %[[t:.*]]: tensor<?x7xf32>, %[[a:.*]]: index, %[[b:.*]]: index
//       CHECK:   %[[bound1:.*]] = affine.apply #[[$map]]()[%[[b]]]
//       CHECK:   %[[c0:.*]] = arith.constant 0 : index
//       CHECK:   %[[dim0:.*]] = tensor.dim %[[t]], %[[c0]]
//       CHECK:   %[[bound0:.*]] = affine.apply #[[$map1]]()[%[[dim0]], %[[a]]]
//       CHECK:   return %[[bound0]], %[[bound1]]
func.func @pad(%t: tensor<?x7xf32>, %a: index, %b: index) -> (index, index) {
  %pad = arith.constant 0.0 : f32
  %0 = tensor.pad %t low[%a, 5] high[%a, %b] {
    ^bb0(%arg1: index, %arg2: index):
      tensor.yield %pad : f32
    } : tensor<?x7xf32> to tensor<?x?xf32>
  %1 = "test.reify_bound"(%0) {dim = 0} : (tensor<?x?xf32>) -> (index)
  %2 = "test.reify_bound"(%0) {dim = 1} : (tensor<?x?xf32>) -> (index)
  return %1, %2 : index, index
}

// -----

// CHECK-LABEL: func @rank(
//  CHECK-SAME:     %[[t:.*]]: tensor<5xf32>
//       CHECK:   %[[c1:.*]] = arith.constant 1 : index
//       CHECK:   return %[[c1]]
func.func @rank(%t: tensor<5xf32>) -> index {
  %0 = tensor.rank %t : tensor<5xf32>
  %1 = "test.reify_bound"(%0) : (index) -> (index)
  return %1 : index
}

// -----

// CHECK: #[[$map:.*]] = affine_map<()[s0, s1] -> (s0 + s1)>
// CHECK-LABEL: func @affine_apply(
//  CHECK-SAME:     %[[a:.*]]: index, %[[b:.*]]: index
//       CHECK:   %[[apply:.*]] = affine.apply #[[$map]]()[%[[a]], %[[b]]]
//       CHECK:   %[[apply:.*]] = affine.apply #[[$map]]()[%[[a]], %[[b]]]
//       CHECL:   return %[[apply]]
func.func @affine_apply(%a: index, %b: index) -> index {
  %0 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%a, %b]
  %1 = "test.reify_bound"(%0) : (index) -> (index)
  return %1 : index
}

// -----

// CHECK-LABEL: func @affine_max_lb(
//  CHECK-SAME:     %[[a:.*]]: index
//       CHECK:   %[[c2:.*]] = arith.constant 2 : index
//       CHECK:   return %[[c2]]
func.func @affine_max_lb(%a: index) -> (index) {
  // Note: There are two LBs: s0 and 2. FlatAffineValueConstraints always
  // returns the constant one at the moment.
  %1 = affine.max affine_map<()[s0] -> (s0, 2)>()[%a]
  %2 = "test.reify_bound"(%1) {type = "LB"}: (index) -> (index)
  return %2 : index
}

// -----

func.func @affine_max_ub(%a: index) -> (index) {
  %1 = affine.max affine_map<()[s0] -> (s0, 2)>()[%a]
  // expected-error @below{{could not reify bound}}
  %2 = "test.reify_bound"(%1) {type = "UB"}: (index) -> (index)
  return %2 : index
}

// -----

// CHECK-LABEL: func @affine_min_ub(
//  CHECK-SAME:     %[[a:.*]]: index
//       CHECK:   %[[c3:.*]] = arith.constant 3 : index
//       CHECK:   return %[[c3]]
func.func @affine_min_ub(%a: index) -> (index) {
  // Note: There are two UBs: s0 + 1 and 3. FlatAffineValueConstraints always
  // returns the constant one at the moment.
  %1 = affine.min affine_map<()[s0] -> (s0, 2)>()[%a]
  %2 = "test.reify_bound"(%1) {type = "UB"}: (index) -> (index)
  return %2 : index
}

// -----

func.func @affine_min_lb(%a: index) -> (index) {
  %1 = affine.min affine_map<()[s0] -> (s0, 2)>()[%a]
  // expected-error @below{{could not reify bound}}
  %2 = "test.reify_bound"(%1) {type = "LB"}: (index) -> (index)
  return %2 : index
}

// -----

// CHECK-LABEL: func @memref_alloc(
//  CHECK-SAME:     %[[sz:.*]]: index
//       CHECK:   %[[c6:.*]] = arith.constant 6 : index
//       CHECK:   return %[[c6]], %[[sz]]
func.func @memref_alloc(%sz: index) -> (index, index) {
  %0 = memref.alloc(%sz) : memref<6x?xf32>
  %1 = "test.reify_bound"(%0) {dim = 0} : (memref<6x?xf32>) -> (index)
  %2 = "test.reify_bound"(%0) {dim = 1} : (memref<6x?xf32>) -> (index)
  return %1, %2 : index, index
}

// -----

// CHECK-LABEL: func @memref_alloca(
//  CHECK-SAME:     %[[sz:.*]]: index
//       CHECK:   %[[c6:.*]] = arith.constant 6 : index
//       CHECK:   return %[[c6]], %[[sz]]
func.func @memref_alloca(%sz: index) -> (index, index) {
  %0 = memref.alloca(%sz) : memref<6x?xf32>
  %1 = "test.reify_bound"(%0) {dim = 0} : (memref<6x?xf32>) -> (index)
  %2 = "test.reify_bound"(%0) {dim = 1} : (memref<6x?xf32>) -> (index)
  return %1, %2 : index, index
}

// -----

// CHECK-LABEL: func @memref_cast(
//       CHECK:   %[[c10:.*]] = arith.constant 10 : index
//       CHECK:   return %[[c10]]
func.func @memref_cast(%m: memref<10xf32>) -> index {
  %0 = memref.cast %m : memref<10xf32> to memref<?xf32>
  %1 = "test.reify_bound"(%0) {dim = 0} : (memref<?xf32>) -> (index)
  return %1 : index
}

// -----

// CHECK-LABEL: func @memref_dim(
//  CHECK-SAME:     %[[m:.*]]: memref<?xf32>
//       CHECK:   %[[dim:.*]] = memref.dim %[[m]]
//       CHECK:   %[[dim:.*]] = memref.dim %[[m]]
//       CHECK:   return %[[dim]]
func.func @memref_dim(%m: memref<?xf32>) -> index {
  %c0 = arith.constant 0 : index
  %0 = memref.dim %m, %c0 : memref<?xf32>
  %1 = "test.reify_bound"(%0) : (index) -> (index)
  return %1 : index
}

// -----

// CHECK-LABEL: func @memref_get_global(
//       CHECK:   %[[c4:.*]] = arith.constant 4 : index
//       CHECK:   return %[[c4]]
memref.global "private" @gv0 : memref<4xf32> = dense<[0.0, 1.0, 2.0, 3.0]>
func.func @memref_get_global() -> index {
  %0 = memref.get_global @gv0 : memref<4xf32>
  %1 = "test.reify_bound"(%0) {dim = 0} : (memref<4xf32>) -> (index)
  return %1 : index
}

// -----

// CHECK-LABEL: func @memref_rank(
//  CHECK-SAME:     %[[t:.*]]: memref<5xf32>
//       CHECK:   %[[c1:.*]] = arith.constant 1 : index
//       CHECK:   return %[[c1]]
func.func @memref_rank(%m: memref<5xf32>) -> index {
  %0 = memref.rank %m : memref<5xf32>
  %1 = "test.reify_bound"(%0) : (index) -> (index)
  return %1 : index
}

// -----

// CHECK-LABEL: func @memref_subview(
//  CHECK-SAME:     %[[m:.*]]: memref<?xf32>, %[[sz:.*]]: index
//       CHECK:   return %[[sz]]
func.func @memref_subview(%m: memref<?xf32>, %sz: index) -> index {
  %0 = memref.subview %m[2][%sz][1] : memref<?xf32> to memref<?xf32, strided<[1], offset: 2>>
  %1 = "test.reify_bound"(%0) {dim = 0} : (memref<?xf32, strided<[1], offset: 2>>) -> (index)
  return %1 : index
}

// -----

// CHECK: #[[$map]] = affine_map<()[s0] -> (s0 + 5)>
// CHECK-LABEL: func @arith_addi(
//  CHECK-SAME:     %[[a:.*]]: index
//       CHECK:   %[[apply:.*]] = affine.apply #[[$map]]()[%[[a]]]
//       CHECK:   return %[[apply]]
func.func @arith_addi(%a: index) -> index {
  %0 = arith.constant 5 : index
  %1 = arith.addi %0, %a : index
  %2 = "test.reify_bound"(%1) : (index) -> (index)
  return %2 : index
}

// -----

// CHECK: #[[$map]] = affine_map<()[s0] -> (-s0 + 5)>
// CHECK-LABEL: func @arith_subi(
//  CHECK-SAME:     %[[a:.*]]: index
//       CHECK:   %[[apply:.*]] = affine.apply #[[$map]]()[%[[a]]]
//       CHECK:   return %[[apply]]
func.func @arith_subi(%a: index) -> index {
  %0 = arith.constant 5 : index
  %1 = arith.subi %0, %a : index
  %2 = "test.reify_bound"(%1) : (index) -> (index)
  return %2 : index
}

// -----

// CHECK: #[[$map]] = affine_map<()[s0] -> (s0 * 5)>
// CHECK-LABEL: func @arith_muli(
//  CHECK-SAME:     %[[a:.*]]: index
//       CHECK:   %[[apply:.*]] = affine.apply #[[$map]]()[%[[a]]]
//       CHECK:   return %[[apply]]
func.func @arith_muli(%a: index) -> index {
  %0 = arith.constant 5 : index
  %1 = arith.muli %0, %a : index
  %2 = "test.reify_bound"(%1) : (index) -> (index)
  return %2 : index
}

// -----

// CHECK-LABEL: func @scf_for(
//  CHECK-SAME:     %[[a:.*]]: index, %[[b:.*]]: index, %[[c:.*]]: index
//       CHECK:   "test.some_use"(%[[a]], %[[b]])
func.func @scf_for(%a: index, %b: index, %c: index) {
  scf.for %iv = %a to %b step %c {
    %0 = "test.reify_bound"(%iv) {type = "LB"} : (index) -> (index)
    %1 = "test.reify_bound"(%iv) {type = "UB"} : (index) -> (index)
    "test.some_use"(%0, %1) : (index, index) -> ()
  }
  return
}

// -----

// CHECK-LABEL: func @linalg_fill(
//  CHECK-SAME:     %[[t:.*]]: tensor<?xf32>
//       CHECK:   %[[c0:.*]] = arith.constant 0 : index
//       CHECK:   %[[dim:.*]] = tensor.dim %[[t]], %[[c0]]
//       CHECK:   return %[[dim]]
func.func @linalg_fill(%t: tensor<?xf32>, %f: f32) -> index {
  %0 = linalg.fill ins(%f : f32) outs(%t : tensor<?xf32>) -> tensor<?xf32>
  %1 = "test.reify_bound"(%0) {dim = 0} : (tensor<?xf32>) -> (index)
  return %1 : index
}
