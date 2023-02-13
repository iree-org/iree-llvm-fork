// RUN: mlir-opt %s -one-shot-bufferize="test-analysis-only" -allow-unregistered-dialect -split-input-file | FileCheck %s

// CHECK-LABEL: func @unknown_op_aliasing(
func.func @unknown_op_aliasing(%f: f32, %f2: f32, %pos: index) -> f32 {
  %0 = tensor.empty() : tensor<10xf32>
  // CHECK: linalg.fill {__inplace_operands_attr__ = ["none", "true"]}
  %1 = linalg.fill ins(%f : f32) outs(%0 : tensor<10xf32>) -> tensor<10xf32>

  // Something must bufferize out-of-place because the op may return an alias
  // of %1.
  // CHECK: "dummy.dummy_op"(%{{.*}}) {__inplace_operands_attr__ = ["false"]}
  %alias = "dummy.dummy_op"(%1) : (tensor<10xf32>) -> (tensor<10xf32>)

  // CHECK: linalg.fill {__inplace_operands_attr__ = ["none", "true"]}
  %2 = linalg.fill ins(%f2 : f32) outs(%1 : tensor<10xf32>) -> tensor<10xf32>
  %3 = tensor.extract %alias[%pos] : tensor<10xf32>
  return %3 : f32
}

// -----

// CHECK-LABEL: func @unknown_op_writing(
func.func @unknown_op_writing(%f: f32, %f2: f32, %pos: index) -> f32 {
  %0 = tensor.empty() : tensor<10xf32>
  // CHECK: linalg.fill {__inplace_operands_attr__ = ["none", "true"]}
  %1 = linalg.fill ins(%f : f32) outs(%0 : tensor<10xf32>) -> tensor<10xf32>

  // The op may bufferize to a memory write, so it must bufferize out-of-place.
  // CHECK: "dummy.dummy_op"(%{{.*}}) {__inplace_operands_attr__ = ["false"]}
  "dummy.dummy_op"(%1) : (tensor<10xf32>) -> ()

  %3 = tensor.extract %1[%pos] : tensor<10xf32>
  return %3 : f32
}

// -----

// CHECK-LABEL: func @read_of_undef_is_not_a_conflict(
func.func @read_of_undef_is_not_a_conflict(%f: f32, %idx: index) -> f32 {
  %0 = tensor.empty() : tensor<10xf32>
  // This can be in-place because the read below does reads undefined data.
  // CHECK: tensor.insert {{.*}} {__inplace_operands_attr__ = ["none", "true", "none"]}
  %1 = tensor.insert %f into %0[%idx] : tensor<10xf32>
  %2 = tensor.extract %0[%idx] : tensor<10xf32>
  return %2 : f32
}

// -----

// CHECK-LABEL: func @read_of_alloc_tensor_is_not_a_conflict(
func.func @read_of_alloc_tensor_is_not_a_conflict(%f: f32, %idx: index) -> f32 {
  %0 = bufferization.alloc_tensor() : tensor<10xf32>
  // This can be in-place because the read below does reads undefined data.
  // CHECK: tensor.insert {{.*}} {__inplace_operands_attr__ = ["none", "true", "none"]}
  %1 = tensor.insert %f into %0[%idx] : tensor<10xf32>
  %2 = tensor.extract %0[%idx] : tensor<10xf32>
  return %2 : f32
}

// -----

// CHECK-LABEL: func @scf_if_definitely_aliasing(
func.func @scf_if_definitely_aliasing(
    %cond: i1, %t1: tensor<?xf32> {bufferization.writable = true},
    %idx: index) -> tensor<?xf32> {
  %r = scf.if %cond -> (tensor<?xf32>) {
    scf.yield %t1 : tensor<?xf32>
  } else {
    // This buffer definitely aliases, so it can be yielded from the block.
    // CHECK: tensor.extract_slice
    // CHECK-SAME: {__inplace_operands_attr__ = ["true", "none", "none"]}
    %t2 = tensor.extract_slice %t1 [%idx] [%idx] [1] : tensor<?xf32> to tensor<?xf32>
    scf.yield %t2 : tensor<?xf32>
  }
  return %r : tensor<?xf32>
}

// -----

// CHECK-LABEL: func @scf_yield_no_allocs(
func.func @scf_yield_no_allocs(
    %c1: i1, %c2: i1, %A: tensor<4xf32>, %B: tensor<4xf32>, %C: tensor<4xf32>)
  -> tensor<4xf32>
{
  %r = scf.if %c1 -> (tensor<4xf32>) {
    // CHECK: arith.select
    // CHECK-SAME: {__inplace_operands_attr__ = ["none", "true", "true"]}
    %0 = arith.select %c2, %A, %B : tensor<4xf32>
    scf.yield %0 : tensor<4xf32>
  } else {
    scf.yield %C : tensor<4xf32>
  }
  return %r: tensor<4xf32>
}

// -----

// Just make sure that bufferization succeeds. The arith.constant ops do not
// allocate.

// CHECK-LABEL: func @block_yield_constant(
func.func @block_yield_constant(%c: i1) -> (tensor<3x4xf32>) {
  %r = scf.if %c -> (tensor<3x4xf32>) {
    %0 = arith.constant dense<7.0> : tensor<3x4xf32>
    scf.yield %0 : tensor<3x4xf32>
  } else {
    %0 = arith.constant dense<8.0> : tensor<3x4xf32>
    scf.yield %0 : tensor<3x4xf32>
  }
  return %r : tensor<3x4xf32>
}
