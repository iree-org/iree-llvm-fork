// RUN: mlir-opt %s -split-input-file -mlir-print-local-scope -test-vector-reorder-transfer -cse | FileCheck %s

func.func @write_as_insert_source(%input: tensor<8x8x8x4xf32>, %val0: vector<4xf32>) -> tensor<8x8x8x4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %extract = tensor.extract_slice %input[4, 3, 1, 2] [1, 4, 4, 4] [1, 1, 1, 1] : tensor<8x8x8x4xf32> to tensor<4x4x4xf32>
  %write0 = vector.transfer_write %val0, %extract[%c1, %c2, %c0] {in_bounds = [true]} : vector<4xf32>, tensor<4x4x4xf32>
  %insert = tensor.insert_slice %write0 into %input[4, 3, 1, 2] [1, 4, 4, 4] [1, 1, 1, 1] : tensor<4x4x4xf32> into tensor<8x8x8x4xf32>
  return %insert : tensor<8x8x8x4xf32>
}

// CHECK-LABEL: func.func @write_as_insert_source
//  CHECK-SAME: (%[[INPUT:.+]]: tensor<8x8x8x4xf32>, %[[VAL:.+]]: vector<4xf32>)
//   CHECK-DAG:   %[[C4:.+]] = arith.constant 4 : index
//   CHECK-DAG:   %[[C3:.+]] = arith.constant 3 : index
//   CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//       CHECK:   %[[WRITE:.+]] = vector.transfer_write %[[VAL]], %[[INPUT]][%[[C4]], %[[C4]], %[[C3]], %[[C2]]]
//  CHECK-SAME:                  {in_bounds = [true]} : vector<4xf32>, tensor<8x8x8x4xf32>
//       CHECK:   return %[[WRITE]]

// -----

// CHECK-LABEL: func.func @write_as_insert_source
//  CHECK-SAME: (%[[INPUT:.+]]: tensor<1x2x2x4xf32>, %[[VAL0:.+]]: vector<4xf32>, %[[VAL1:.+]]: vector<4xf32>)
//       CHECK:   %[[W0:.+]] = vector.transfer_write %[[VAL0]], %[[INPUT]]
//       CHECK:   %[[W1:.+]] = vector.transfer_write %[[VAL1]], %[[W0]]
//       CHECK:   return %[[W1]]
func.func @write_as_insert_source(%input: tensor<1x2x2x4xf32>, %val0: vector<4xf32>, %val1: vector<4xf32>) -> tensor<1x2x2x4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %extract = tensor.extract_slice %input[0, 0, 0, 0] [1, 1, 2, 4] [1, 1, 1, 1] : tensor<1x2x2x4xf32> to tensor<1x2x4xf32>
  %write0 = vector.transfer_write %val0, %extract[%c0, %c0, %c0] {in_bounds = [true]} : vector<4xf32>, tensor<1x2x4xf32>
  %write1 = vector.transfer_write %val1, %write0[%c0, %c1, %c0] {in_bounds = [true]} : vector<4xf32>, tensor<1x2x4xf32>
  %insert = tensor.insert_slice %write1 into %input[%c0, 0, %c0, 0] [1, 1, 2, 4] [1, 1, 1, 1] : tensor<1x2x4xf32> into tensor<1x2x2x4xf32>
  return %insert : tensor<1x2x2x4xf32>
}

// -----

// CHECK-LABEL: func.func @transfer_not_straddling_reduced_ranks
//  CHECK-SAME: (%[[INPUT:.+]]: tensor<1x?x?x?xf32>, %[[I1:.+]]: index, %[[I2:.+]]: index, %[[I3:.+]]: index, %[[I4:.+]]: index, %[[I5:.+]]: index, %[[VAL:.+]]: vector<4xf32>)
//       CHECK:   %[[C0:.+]] = arith.constant 0 : index
//       CHECK:   %[[IDX1:.+]] = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%[[I4]], %[[I1]]]
//       CHECK:   %[[IDX3:.+]] = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%[[I5]], %[[I3]]]
//       CHECK:   %[[W:.+]] = vector.transfer_write %[[VAL]], %[[INPUT]][%[[C0]], %[[IDX1]], %[[I2]], %[[IDX3]]]
//       CHECK:   return %[[W]]
func.func @transfer_not_straddling_reduced_ranks(
    %input: tensor<1x?x?x?xf32>, %i1: index, %i2: index, %i3: index, %i4: index, %i5: index, %val0: vector<4xf32>) -> tensor<1x?x?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %extract = tensor.extract_slice %input[0, %i1, %i2, %i3] [1, 2, 1, 4] [1, 1, 1, 1] : tensor<1x?x?x?xf32> to tensor<1x2x4xf32>
  // We can swap this case given that the result dim#0 of the transfer_write maps to
  // dim#3 of the extract/insert slice source tensor; the innermost trimmed unit
  // dim of extract/insert slice is dim#2.
  %write0 = vector.transfer_write %val0, %extract[%c0, %i4, %i5] {in_bounds = [true]} : vector<4xf32>, tensor<1x2x4xf32>
  %insert = tensor.insert_slice %write0 into %input[%c0, %i1, %i2, %i3] [1, 2, 1, 4] [1, 1, 1, 1] : tensor<1x2x4xf32> into tensor<1x?x?x?xf32>
  return %insert : tensor<1x?x?x?xf32>
}

// -----

//   CHECK-LABEL: func.func @transfer_straddling_reduced_ranks
//         CHECK:   tensor.extract_slice
//         CHECK:   vector.transfer_write
//         CHECK:   tensor.insert_slice
func.func @transfer_straddling_reduced_ranks(
    %input: tensor<1x?x?x?xf32>, %i1: index, %i2: index, %i3: index, %i4: index, %i5: index, %val0: vector<2x4xf32>) -> tensor<1x?x?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %extract = tensor.extract_slice %input[0, %i1, %i2, %i3] [1, 2, 1, 4] [1, 1, 1, 1] : tensor<1x?x?x?xf32> to tensor<1x2x4xf32>
  // We cannot swap this case because result dim#0 of the transfer_write maps to
  // dim#1 of the extract/insert slice source tensor; but dim#2 is a trimmed unit dim.
  %write0 = vector.transfer_write %val0, %extract[%c0, %i4, %i5] {in_bounds = [true, true]} : vector<2x4xf32>, tensor<1x2x4xf32>
  %insert = tensor.insert_slice %write0 into %input[%c0, %i1, %i2, %i3] [1, 2, 1, 4] [1, 1, 1, 1] : tensor<1x2x4xf32> into tensor<1x?x?x?xf32>
  return %insert : tensor<1x?x?x?xf32>
}

// -----

//   CHECK-LABEL: func.func @not_minor_identity_map
//         CHECK: tensor.extract_slice
// CHECK-COUNT-2: vector.transfer_write
//         CHECK: tensor.insert_slice
func.func @not_minor_identity_map(%input: tensor<1x2x2x4xf32>, %val0: vector<2xf32>, %val1: vector<2xf32>) -> tensor<1x2x2x4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %extract = tensor.extract_slice %input[0, 0, 0, 0] [1, 1, 2, 4] [1, 1, 1, 1] : tensor<1x2x2x4xf32> to tensor<1x2x4xf32>
  %write0 = vector.transfer_write %val0, %extract[%c0, %c0, %c0] {in_bounds = [true], permutation_map = affine_map<(d0, d1, d2) -> (d1)>} : vector<2xf32>, tensor<1x2x4xf32>
  %write1 = vector.transfer_write %val1, %write0[%c0, %c1, %c0] {in_bounds = [true], permutation_map = affine_map<(d0, d1, d2) -> (d1)>} : vector<2xf32>, tensor<1x2x4xf32>
  %insert = tensor.insert_slice %write1 into %input[%c0, 0, %c0, 0] [1, 1, 2, 4] [1, 1, 1, 1] : tensor<1x2x4xf32> into tensor<1x2x2x4xf32>
  return %insert : tensor<1x2x2x4xf32>
}

// -----

//   CHECK-LABEL: func.func @mismatched_slice_parameters
//         CHECK: tensor.extract_slice
// CHECK-COUNT-2: vector.transfer_write
//         CHECK: tensor.insert_slice
func.func @mismatched_slice_parameters(%input: tensor<1x2x2x4xf32>, %val0: vector<4xf32>, %val1: vector<4xf32>) -> tensor<1x2x2x4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %extract = tensor.extract_slice %input[0, 0, 0, 0] [1, 1, 2, 4] [1, 1, 1, 1] : tensor<1x2x2x4xf32> to tensor<1x2x4xf32>
  %write0 = vector.transfer_write %val0, %extract[%c0, %c0, %c0] {in_bounds = [true]} : vector<4xf32>, tensor<1x2x4xf32>
  %write1 = vector.transfer_write %val1, %write0[%c0, %c1, %c0] {in_bounds = [true]} : vector<4xf32>, tensor<1x2x4xf32>
  %insert = tensor.insert_slice %write1 into %input[%c0, 1, %c0, 0] [1, 1, 2, 4] [1, 1, 1, 1] : tensor<1x2x4xf32> into tensor<1x2x2x4xf32>
  return %insert : tensor<1x2x2x4xf32>
}

// -----

//   CHECK-LABEL: func.func @not_insert_back_to_original_tensor
//         CHECK: tensor.extract_slice
// CHECK-COUNT-2: vector.transfer_write
//         CHECK: tensor.insert_slice
func.func @not_insert_back_to_original_tensor(%input0: tensor<1x2x2x4xf32>, %input1: tensor<1x2x2x4xf32>, %val0: vector<4xf32>, %val1: vector<4xf32>) -> tensor<1x2x2x4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %extract = tensor.extract_slice %input0[0, 0, 0, 0] [1, 1, 2, 4] [1, 1, 1, 1] : tensor<1x2x2x4xf32> to tensor<1x2x4xf32>
  %write0 = vector.transfer_write %val0, %extract[%c0, %c0, %c0] {in_bounds = [true]} : vector<4xf32>, tensor<1x2x4xf32>
  %write1 = vector.transfer_write %val1, %write0[%c0, %c1, %c0] {in_bounds = [true]} : vector<4xf32>, tensor<1x2x4xf32>
  %insert = tensor.insert_slice %write1 into %input1[%c0, 0, %c0, 0] [1, 1, 2, 4] [1, 1, 1, 1] : tensor<1x2x4xf32> into tensor<1x2x2x4xf32>
  return %insert : tensor<1x2x2x4xf32>
}

// -----

// CHECK-LABEL: func.func @write_as_insert_dest
//  CHECK-SAME: (%[[INPUT:.+]]: tensor<1x2x2x4xf32>, %[[VAL0:.+]]: vector<4xf32>, %[[VAL1:.+]]: vector<4xf32>)
//       CHECK:   %[[C0:.+]] = arith.constant 0 : index
//       CHECK:   %[[C1:.+]] = arith.constant 1 : index
//       CHECK:   %[[W0:.+]] = vector.transfer_write %[[VAL0]], %[[INPUT]][%[[C0]], %[[C0]], %[[C0]], %[[C0]]] {in_bounds = [true]} : vector<4xf32>, tensor<1x2x2x4xf32>
//       CHECK:   %[[W1:.+]] = vector.transfer_write %[[VAL1]], %[[W0]][%[[C0]], %[[C0]], %[[C1]], %[[C0]]] {in_bounds = [true]} : vector<4xf32>, tensor<1x2x2x4xf32>
//       CHECK:   return %[[W1]] : tensor<1x2x2x4xf32>
func.func @write_as_insert_dest(%input: tensor<1x2x2x4xf32>, %val0: vector<4xf32>, %val1: vector<4xf32>) -> tensor<1x2x2x4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = tensor.extract_slice %input[0, 1, 0, 0] [1, 1, 2, 4] [1, 1, 1, 1] : tensor<1x2x2x4xf32> to tensor<1x2x4xf32>
  %3 = vector.transfer_write %val0, %input[%c0, %c0, %c0, %c0] {in_bounds = [true]} : vector<4xf32>, tensor<1x2x2x4xf32>
  %4 = vector.transfer_write %val1, %3[%c0, %c0, %c1, %c0] {in_bounds = [true]} : vector<4xf32>, tensor<1x2x2x4xf32>
  %5 = tensor.insert_slice %0 into %4[0, 1, 0, 0] [1, 1, 2, 4] [1, 1, 1, 1] : tensor<1x2x4xf32> into tensor<1x2x2x4xf32>
return %5 : tensor<1x2x2x4xf32>
}

// -----

// CHECK-LABEL: func.func @not_disjoint_ranges
//       CHECK:   %[[W:.+]] = vector.transfer_write
//       CHECK:   tensor.insert_slice %{{.+}} into %[[W]]
func.func @not_disjoint_ranges(%input: tensor<1x2x2x4xf32>, %val: vector<4xf32>, %slice: tensor<1x2x4xf32>) -> tensor<1x2x2x4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = vector.transfer_write %val, %input[%c0, %c1, %c0, %c0] {in_bounds = [true]} : vector<4xf32>, tensor<1x2x2x4xf32>
  %1 = tensor.insert_slice %slice into %0[0, 1, 0, 0] [1, 1, 2, 4] [1, 1, 1, 1] : tensor<1x2x4xf32> into tensor<1x2x2x4xf32>
return %1 : tensor<1x2x2x4xf32>
}

// -----

// CHECK-LABEL: func.func @not_minor_identity_map
//       CHECK:   %[[W:.+]] = vector.transfer_write
//       CHECK:   tensor.insert_slice %{{.+}} into %[[W]]
func.func @not_minor_identity_map(%input: tensor<1x2x2x4xf32>, %val: vector<2xf32>, %slice: tensor<1x2x4xf32>) -> tensor<1x2x2x4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = vector.transfer_write %val, %input[%c0, %c0, %c0, %c0] {in_bounds = [true], permutation_map = affine_map<(d0, d1, d2, d3) -> (d2)>} : vector<2xf32>, tensor<1x2x2x4xf32>
  %1 = tensor.insert_slice %slice into %0[0, 1, 0, 0] [1, 1, 2, 4] [1, 1, 1, 1] : tensor<1x2x4xf32> into tensor<1x2x2x4xf32>
return %1 : tensor<1x2x2x4xf32>
}

// -----

// CHECK-LABEL: func.func @rank_reduced_not_disjoint
//       CHECK:   %[[W:.+]] = vector.transfer_write
//       CHECK:   tensor.insert_slice %{{.+}} into %[[W]]
func.func @rank_reduced_not_disjoint(
    %input: tensor<1x2x2x4xf32>, %i1: index, %i2: index, %val: vector<4xf32>, %slice: tensor<1x2x4xf32>) -> tensor<1x2x2x4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = vector.transfer_write %val, %input[%c0, %i1, %i2, %c0] {in_bounds = [true]} : vector<4xf32>, tensor<1x2x2x4xf32>
  %1 = tensor.insert_slice %slice into %0[0, %i1, %i2, 0] [1, 2, 1, 4] [1, 1, 1, 1] : tensor<1x2x4xf32> into tensor<1x2x2x4xf32>
  return %1 : tensor<1x2x2x4xf32>
}

// -----

// CHECK-LABEL: func.func @rank_reduced_disjoint
//  CHECK-SAME: (%[[INPUT:.+]]: tensor<1x2x2x4xf32>, %[[I1:.+]]: index, %[[VAL:.+]]: vector<4xf32>, %[[SLICE:.+]]: tensor<1x2x4xf32>)
//       CHECK:   %[[C0:.+]] = arith.constant 0 : index
//       CHECK:   %[[INSERT:.+]] = tensor.insert_slice %[[SLICE]] into %[[INPUT]][0, %[[I1]], 1, 0] [1, 2, 1, 4] [1, 1, 1, 1]
//       CHECK:   %[[W:.+]] = vector.transfer_write %[[VAL]], %[[INSERT]][%[[C0]], %[[I1]], %[[C0]], %[[C0]]]
//       CHECK:   return %[[W]] : tensor<1x2x2x4xf32>
func.func @rank_reduced_disjoint(
    %input: tensor<1x2x2x4xf32>, %i1: index, %val: vector<4xf32>, %slice: tensor<1x2x4xf32>) -> tensor<1x2x2x4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = vector.transfer_write %val, %input[%c0, %i1, %c0, %c0] {in_bounds = [true]} : vector<4xf32>, tensor<1x2x2x4xf32>
  %1 = tensor.insert_slice %slice into %0[0, %i1, %c1, 0] [1, 2, 1, 4] [1, 1, 1, 1] : tensor<1x2x4xf32> into tensor<1x2x2x4xf32>
  return %1 : tensor<1x2x2x4xf32>
}

// -----

// CHECK-LABEL: func.func @two_extract_insert_groups
//  CHECK-SAME: (%[[INPUT:.+]]: tensor<8x2x2x4xf32>, %[[VAL0:.+]]: vector<4xf32>, %[[VAL1:.+]]: vector<4xf32>, %[[VAL2:.+]]: vector<4xf32>, %[[VAL3:.+]]: vector<4xf32>)
//   CHECK-DAG:   %[[C6:.+]] = arith.constant 6 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C4:.+]] = arith.constant 4 : index
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//       CHECK:   %[[W0:.+]] = vector.transfer_write %[[VAL2]], %[[INPUT]][%[[C6]], %[[C1]], %[[C0]], %[[C0]]]
//       CHECK:   %[[W1:.+]] = vector.transfer_write %[[VAL3]], %[[W0]][%[[C6]], %[[C1]], %[[C1]], %[[C0]]]
//       CHECK:   %[[W2:.+]] = vector.transfer_write %[[VAL0]], %[[W1]][%[[C4]], %[[C0]], %[[C0]], %[[C0]]]
//       CHECK:   %[[W3:.+]] = vector.transfer_write %[[VAL1]], %[[W2]][%[[C4]], %[[C0]], %[[C1]], %c0]
//       CHECK:   return %[[W3]]
func.func @two_extract_insert_groups(%input: tensor<8x2x2x4xf32>, %val0: vector<4xf32>, %val1: vector<4xf32>, %val2: vector<4xf32>, %val3: vector<4xf32>) -> tensor<8x2x2x4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = tensor.extract_slice %input[6, %c1, %c0, %c0] [1, 1, 2, 4] [1, 1, 1, 1] : tensor<8x2x2x4xf32> to tensor<1x2x4xf32>
  %1 = tensor.extract_slice %input[4, %c0, %c0, %c0] [1, 1, 2, 4] [1, 1, 1, 1] : tensor<8x2x2x4xf32> to tensor<1x2x4xf32>
  %2 = vector.transfer_write %val0, %1[%c0, %c0, %c0] {in_bounds = [true]} : vector<4xf32>, tensor<1x2x4xf32>
  %3 = vector.transfer_write %val1, %2[%c0, %c1, %c0] {in_bounds = [true]} : vector<4xf32>, tensor<1x2x4xf32>
  %4 = vector.transfer_write %val2, %0[%c0, %c0, %c0] {in_bounds = [true]} : vector<4xf32>, tensor<1x2x4xf32>
  %5 = vector.transfer_write %val3, %4[%c0, %c1, %c0] {in_bounds = [true]} : vector<4xf32>, tensor<1x2x4xf32>
  %6 = tensor.insert_slice %3 into %input[4, 0, 0, 0] [1, 1, 2, 4] [1, 1, 1, 1] : tensor<1x2x4xf32> into tensor<8x2x2x4xf32>
  %7 = tensor.insert_slice %5 into %6[6, 1, 0, 0] [1, 1, 2, 4] [1, 1, 1, 1] : tensor<1x2x4xf32> into tensor<8x2x2x4xf32>
  return %7 : tensor<8x2x2x4xf32>
}

// -----

// CHECK-LABEL: func.func @two_levels_of_extract_insert
//  CHECK-SAME: (%[[INPUT:.+]]: tensor<2x4x16x16xf32>, %[[I2:.+]]: index, %[[I3:.+]]: index,
//  CHECK-SAME:  %[[VAL0:.+]]: vector<4xf32>, %[[VAL1:.+]]: vector<4xf32>, %[[VAL2:.+]]: vector<4xf32>, %[[VAL3:.+]]: vector<4xf32>)
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//  CHECK-DAG:   %[[C3:.+]] = arith.constant 3 : index
//      CHECK:   %[[IDX2:.+]] = affine.apply affine_map<()[s0] -> (s0 + 1)>()[%[[I2]]]
//      CHECK:   %[[W2:.+]] = vector.transfer_write %[[VAL2]], %[[INPUT]][%[[C0]], %[[C2]], %[[IDX2]], %[[I3]]]
//      CHECK:   %[[W3:.+]] = vector.transfer_write %[[VAL3]], %[[W2]][%[[C1]], %[[C3]], %[[IDX2]], %[[I3]]]
//      CHECK:   %[[W0:.+]] = vector.transfer_write %[[VAL0]], %[[W3]][%[[C0]], %[[C2]], %[[I2]], %[[I3]]]
//      CHECK:   %[[W1:.+]] = vector.transfer_write %[[VAL1]], %[[W0]][%[[C1]], %[[C3]], %[[I2]], %[[I3]]]
//      CHECK:   return %[[W1]] : tensor<2x4x16x16xf32>
func.func @two_levels_of_extract_insert(
    %input: tensor<2x4x16x16xf32>, %i2: index, %i3: index,
    %val0: vector<4xf32>, %val1: vector<4xf32>, %val2: vector<4xf32>, %val3: vector<4xf32>) -> tensor<2x4x16x16xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index

  %0 = tensor.extract_slice %input[0, 0, %i2, %i3] [2, 4, 2, 4] [1, 1, 1, 1] : tensor<2x4x16x16xf32> to tensor<2x4x2x4xf32>
  %1 = tensor.extract_slice %0[0, 0, 1, 0] [2, 4, 1, 4] [1, 1, 1, 1] : tensor<2x4x2x4xf32> to tensor<2x4x4xf32>
  %2 = tensor.extract_slice %0[0, 0, 0, 0] [2, 4, 1, 4] [1, 1, 1, 1] : tensor<2x4x2x4xf32> to tensor<2x4x4xf32>

  %3 = vector.transfer_write %val0, %2[%c0, %c2, %c0] {in_bounds = [true]} : vector<4xf32>, tensor<2x4x4xf32>
  %4 = vector.transfer_write %val1, %3[%c1, %c3, %c0] {in_bounds = [true]} : vector<4xf32>, tensor<2x4x4xf32>

  %5 = vector.transfer_write %val2, %1[%c0, %c2, %c0] {in_bounds = [true]} : vector<4xf32>, tensor<2x4x4xf32>
  %6 = vector.transfer_write %val3, %5[%c1, %c3, %c0] {in_bounds = [true]} : vector<4xf32>, tensor<2x4x4xf32>

  %7 = tensor.insert_slice %4 into %0[0, 0, 0, 0] [2, 4, 1, 4] [1, 1, 1, 1] : tensor<2x4x4xf32> into tensor<2x4x2x4xf32>
  %8 = tensor.insert_slice %6 into %7[0, 0, 1, 0] [2, 4, 1, 4] [1, 1, 1, 1] : tensor<2x4x4xf32> into tensor<2x4x2x4xf32>
  %9 = tensor.insert_slice %8 into %input[0, 0, %i2, %i3] [2, 4, 2, 4] [1, 1, 1, 1] : tensor<2x4x2x4xf32> into tensor<2x4x16x16xf32>
  return %9: tensor<2x4x16x16xf32>
}
