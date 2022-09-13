// RUN: mlir-opt -split-input-file -test-tensor-transform-patterns=test-extract-from-insert-slice-dest -canonicalize %s | FileCheck %s

func.func @disjoint_insert_extract_slice_static_shape(%src: tensor<1x2x4xf32>, %dst: tensor<1x2x2x4xf32>) -> tensor<1x2x4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %insert = tensor.insert_slice %src into %dst[0, 0, 0, 0] [1, 1, 2, 4] [1, 1, 1, 1] : tensor<1x2x4xf32> into tensor<1x2x2x4xf32>
  %extract = tensor.extract_slice %insert[%c0, %c1, %c0, %c0] [1, 1, 2, 4] [1, 1, 1, 1] : tensor<1x2x2x4xf32> to tensor<1x2x4xf32>
  return %extract : tensor<1x2x4xf32>
}

// CHECK-LABEL: func.func @disjoint_insert_extract_slice_static_shape
//  CHECK-SAME: (%{{.+}}: tensor<1x2x4xf32>, %[[DST:.+]]: tensor<1x2x2x4xf32>)
// CHECK:   %[[EXTRACT:.+]] = tensor.extract_slice %[[DST]][0, 1, 0, 0] [1, 1, 2, 4] [1, 1, 1, 1]
// CHECK:   return %[[EXTRACT]]

// -----

func.func @disjoint_insert_extract_slice_static_shape(%src: tensor<1x2x4xf32>, %dst: tensor<1x2x2x4xf32>) -> tensor<1x2x4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %insert = tensor.insert_slice %src into %dst[0, 1, 0, 0] [1, 1, 2, 4] [1, 1, 1, 1] : tensor<1x2x4xf32> into tensor<1x2x2x4xf32>
  %extract = tensor.extract_slice %insert[%c0, %c0, %c0, %c0] [1, 1, 2, 4] [1, 1, 1, 1] : tensor<1x2x2x4xf32> to tensor<1x2x4xf32>
  return %extract : tensor<1x2x4xf32>
}

// CHECK-LABEL: func.func @disjoint_insert_extract_slice_static_shape
//  CHECK-SAME: (%{{.+}}: tensor<1x2x4xf32>, %[[DST:.+]]: tensor<1x2x2x4xf32>)
//       CHECK:   %[[EXTRACT:.+]] = tensor.extract_slice %[[DST]][0, 0, 0, 0] [1, 1, 2, 4] [1, 1, 1, 1]
//       CHECK:   return %[[EXTRACT]]

// -----

func.func @disjoint_insert_extract_slice_dynamic_shape(%src: tensor<2x?xf32>, %dst: tensor<8x?xf32>, %size: index) -> tensor<3x?xf32> {
  %insert = tensor.insert_slice %src into %dst[2, 0] [2, %size] [1, 1] : tensor<2x?xf32> into tensor<8x?xf32>
  %extract = tensor.extract_slice %insert[5, 0] [3, %size] [1, 1] : tensor<8x?xf32> to tensor<3x?xf32>
  return %extract : tensor<3x?xf32>
}

// CHECK-LABEL: func.func @disjoint_insert_extract_slice_dynamic_shape
//  CHECK-SAME: (%{{.+}}: tensor<2x?xf32>, %[[DST:.+]]: tensor<8x?xf32>, %[[SIZE:.+]]: index)
//       CHECK:   %[[EXTRACT:.+]] = tensor.extract_slice %[[DST]][5, 0] [3, %[[SIZE]]] [1, 1]
//       CHECK:   return %[[EXTRACT]]

// -----

func.func @joint_insert_extract_slice_dynamic_shape(%src: tensor<2x?xf32>, %dst: tensor<8x?xf32>, %size: index) -> tensor<3x?xf32> {
  %insert = tensor.insert_slice %src into %dst[2, 0] [2, %size] [1, 1] : tensor<2x?xf32> into tensor<8x?xf32>
  %extract = tensor.extract_slice %insert[3, 0] [3, %size] [1, 1] : tensor<8x?xf32> to tensor<3x?xf32>
  return %extract : tensor<3x?xf32>
}

// CHECK-LABEL: func.func @joint_insert_extract_slice_dynamic_shape
//       CHECK:   tensor.insert_slice
//       CHECK:   tensor.extract_slice

// -----

func.func @joint_insert_extract_slice_dynamic_shape(%src: tensor<2x?xf32>, %dst: tensor<8x?xf32>, %size: index) -> tensor<3x?xf32> {
  %insert = tensor.insert_slice %src into %dst[2, 0] [2, %size] [1, 1] : tensor<2x?xf32> into tensor<8x?xf32>
  %extract = tensor.extract_slice %insert[1, 0] [3, %size] [1, 1] : tensor<8x?xf32> to tensor<3x?xf32>
  return %extract : tensor<3x?xf32>
}

// CHECK-LABEL: func.func @joint_insert_extract_slice_dynamic_shape
//       CHECK:   tensor.insert_slice
//       CHECK:   tensor.extract_slice


// -----

func.func @joint_insert_extract_slice_dynamic_shape(%src: tensor<2x?xf32>, %dst: tensor<8x?xf32>, %offset: index, %size: index) -> tensor<3x?xf32> {
  %insert = tensor.insert_slice %src into %dst[2, 0] [2, %size] [1, 1] : tensor<2x?xf32> into tensor<8x?xf32>
  %extract = tensor.extract_slice %insert[%offset, 0] [3, %size] [1, 1] : tensor<8x?xf32> to tensor<3x?xf32>
  return %extract : tensor<3x?xf32>
}

// CHECK-LABEL: func.func @joint_insert_extract_slice_dynamic_shape
//       CHECK:   tensor.insert_slice
//       CHECK:   tensor.extract_slice
