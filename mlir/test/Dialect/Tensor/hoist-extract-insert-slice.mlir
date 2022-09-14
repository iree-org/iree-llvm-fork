// RUN: mlir-opt -split-input-file -test-tensor-transform-patterns=test-hoist-extract-insert-slice -allow-unregistered-dialect -canonicalize %s | FileCheck %s

func.func @hoist_slices_in_double_loop(
    %input: tensor<1x9x9x3xf32>, %filter: tensor<3x3x3x16xf32>, %init: tensor<1x2x2x4xf32>,
    %offset0: index, %offset1: index, %offset2: index) -> tensor<1x2x2x4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %0 = scf.for %iv0 = %c0 to %c3 step %c1 iter_args(%arg0 = %init) -> (tensor<1x2x2x4xf32>) {
    %1 = scf.for %iv1 = %c0 to %c3 step %c1 iter_args(%arg1 = %arg0) -> (tensor<1x2x2x4xf32>) {
      %2 = affine.apply affine_map<(d0, d1) -> (d0 * 2 + d1)>(%offset0, %iv0)
      %3 = affine.apply affine_map<(d0, d1) -> (d0 * 2 + d1)>(%offset1, %iv1)
      %4 = tensor.extract_slice %input[%c0, %2, %3, %c0] [1, 1, 3, 3] [1, 1, 1, 1] : tensor<1x9x9x3xf32> to tensor<1x3x3xf32>
      %5 = tensor.extract_slice %filter[%iv0, %iv1, %c0, %offset2] [1, 1, 3, 4] [1, 1, 1, 1] : tensor<3x3x3x16xf32> to tensor<1x3x4xf32>
      %6 = tensor.extract_slice %arg1[%c0, %c0, %c0, %c0] [1, 1, 2, 4] [1, 1, 1, 1] : tensor<1x2x2x4xf32> to tensor<1x2x4xf32>
      %7 = linalg.conv_1d_nwc_wcf {dilations = dense<1> : vector<1xi64>, strides = dense<2> : vector<1xi64>}
            ins(%4, %5 : tensor<1x3x3xf32>, tensor<1x3x4xf32>) outs(%6 : tensor<1x2x4xf32>) -> tensor<1x2x4xf32>
      %8 = tensor.insert_slice %7 into %arg1[0, 0, 0, 0] [1, 1, 2, 4] [1, 1, 1, 1] : tensor<1x2x4xf32> into tensor<1x2x2x4xf32>
      %9 = affine.apply affine_map<(d0, d1) -> (d0 * 2 + d1 + 2)>(%offset0, %iv0)
      %10 = affine.apply affine_map<(d0, d1) -> (d0 * 2 + d1)>(%offset1, %iv1)
      %11 = tensor.extract_slice %input[%c0, %9, %10, %c0] [1, 1, 3, 3] [1, 1, 1, 1] : tensor<1x9x9x3xf32> to tensor<1x3x3xf32>
      %12 = tensor.extract_slice %filter[%iv0, %iv1, %c0, %offset2] [1, 1, 3, 4] [1, 1, 1, 1] : tensor<3x3x3x16xf32> to tensor<1x3x4xf32>
      %13 = tensor.extract_slice %arg1[%c0, %c1, %c0, %c0] [1, 1, 2, 4] [1, 1, 1, 1] : tensor<1x2x2x4xf32> to tensor<1x2x4xf32>
      %14 = linalg.conv_1d_nwc_wcf {dilations = dense<1> : vector<1xi64>, strides = dense<2> : vector<1xi64>}
            ins(%11, %12 : tensor<1x3x3xf32>, tensor<1x3x4xf32>) outs(%13 : tensor<1x2x4xf32>) -> tensor<1x2x4xf32>
      %15 = tensor.insert_slice %14 into %8[0, 1, 0, 0] [1, 1, 2, 4] [1, 1, 1, 1] : tensor<1x2x4xf32> into tensor<1x2x2x4xf32>
      scf.yield %15 : tensor<1x2x2x4xf32>
    }
    scf.yield %1 : tensor<1x2x2x4xf32>
  }
  return %0 : tensor<1x2x2x4xf32>
}

// CHECK-LABEL: func.func @hoist_slices_in_double_loop
//  CHECK-SAME: (%[[INPUT:.+]]: tensor<1x9x9x3xf32>, %[[FILTER:.+]]: tensor<3x3x3x16xf32>, %[[INIT:.+]]: tensor<1x2x2x4xf32>
//       CHECK:   %[[C0:.+]] = arith.constant 0 : index
//       CHECK:   %[[C1:.+]] = arith.constant 1 : index
//       CHECK:   %[[C3:.+]] = arith.constant 3 : index
//       CHECK:   %[[INIT_SLICE1:.+]] = tensor.extract_slice %[[INIT]][0, 1, 0, 0] [1, 1, 2, 4] [1, 1, 1, 1] : tensor<1x2x2x4xf32> to tensor<1x2x4xf32>
//       CHECK:   %[[INIT_SLICE0:.+]] = tensor.extract_slice %[[INIT]][0, 0, 0, 0] [1, 1, 2, 4] [1, 1, 1, 1] : tensor<1x2x2x4xf32> to tensor<1x2x4xf32>
//       CHECK:   %[[FOR0:.+]]:2 = scf.for %[[IV0:.+]] = %[[C0]] to %[[C3]] step %[[C1]]
//  CHECK-SAME:                      iter_args(%[[FOR0_ARG1:.+]] = %[[INIT_SLICE1]], %[[FOR0_ARG0:.+]] = %[[INIT_SLICE0]])
//       CHECK:     %[[FOR1:.+]]:2 = scf.for %[[IV1:.+]] = %[[C0]] to %[[C3]] step %[[C1]]
//  CHECK-SAME:                        iter_args(%[[FOR1_ARG1:.+]] = %[[FOR0_ARG1]], %[[FOR1_ARG0:.+]] = %[[FOR0_ARG0]])
//       CHECK:       %[[INPUT_SLICE0:.+]] = tensor.extract_slice %[[INPUT]]
//       CHECK:       %[[FILTER_SLICE0:.+]] = tensor.extract_slice %[[FILTER]]
//       CHECK:       %[[CONV0:.+]] = linalg.conv_1d_nwc_wcf
//  CHECK-SAME:                         ins(%[[INPUT_SLICE0]], %[[FILTER_SLICE0]]
//  CHECK-SAME:                         outs(%[[FOR1_ARG0]] : tensor<1x2x4xf32>)
//       CHECK:       %[[INPUT_SLICE1:.+]] = tensor.extract_slice %[[INPUT]]
//       CHECK:       %[[FILTER_SLICE1:.+]] = tensor.extract_slice %[[FILTER]]
//       CHECK:       %[[CONV1:.+]] = linalg.conv_1d_nwc_wcf
//  CHECK-SAME:                         ins(%[[INPUT_SLICE1]], %[[FILTER_SLICE1]]
//  CHECK-SAME:                         outs(%[[FOR1_ARG1]] : tensor<1x2x4xf32>)
//       CHECK:       scf.yield %[[CONV1]], %[[CONV0]] : tensor<1x2x4xf32>, tensor<1x2x4xf32>
//       CHECK:     }
//       CHECK:     scf.yield %[[FOR1]]#0, %[[FOR1]]#1 : tensor<1x2x4xf32>, tensor<1x2x4xf32>
//       CHECK:   }
//       CHECK:   %[[INSERT0:.+]] = tensor.insert_slice %[[FOR0]]#1 into %[[INIT]][0, 0, 0, 0] [1, 1, 2, 4] [1, 1, 1, 1] : tensor<1x2x4xf32> into tensor<1x2x2x4xf32>
//       CHECK:   %[[INSERT1:.+]] = tensor.insert_slice %[[FOR0]]#0 into %[[INSERT0]][0, 1, 0, 0] [1, 1, 2, 4] [1, 1, 1, 1] : tensor<1x2x4xf32> into tensor<1x2x2x4xf32>
//       CHECK:   return %[[INSERT1]] : tensor<1x2x2x4xf32>

// -----

func.func @dont_hoist_non_extract_insert_slice_usage_of_loop_carried_value(
    %input: tensor<1x9x9x3xf32>, %filter: tensor<3x3x3x16xf32>, %init: tensor<1x2x2x4xf32>,
    %offset0: index, %offset1: index, %offset2: index) -> tensor<1x2x2x4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %0 = scf.for %iv0 = %c0 to %c3 step %c1 iter_args(%arg0 = %init) -> (tensor<1x2x2x4xf32>) {
    %1 = scf.for %iv1 = %c0 to %c3 step %c1 iter_args(%arg1 = %arg0) -> (tensor<1x2x2x4xf32>) {
      %2 = affine.apply affine_map<(d0, d1) -> (d0 * 2 + d1)>(%offset0, %iv0)
      %3 = affine.apply affine_map<(d0, d1) -> (d0 * 2 + d1)>(%offset1, %iv1)
      %4 = tensor.extract_slice %input[%c0, %2, %3, %c0] [1, 1, 3, 3] [1, 1, 1, 1] : tensor<1x9x9x3xf32> to tensor<1x3x3xf32>
      %5 = tensor.extract_slice %filter[%iv0, %iv1, %c0, %offset2] [1, 1, 3, 4] [1, 1, 1, 1] : tensor<3x3x3x16xf32> to tensor<1x3x4xf32>
      %6 = tensor.extract_slice %arg1[%c0, %c0, %c0, %c0] [1, 1, 2, 4] [1, 1, 1, 1] : tensor<1x2x2x4xf32> to tensor<1x2x4xf32>
      %7 = linalg.conv_1d_nwc_wcf {dilations = dense<1> : vector<1xi64>, strides = dense<2> : vector<1xi64>}
            ins(%4, %5 : tensor<1x3x3xf32>, tensor<1x3x4xf32>) outs(%6 : tensor<1x2x4xf32>) -> tensor<1x2x4xf32>
      %8 = tensor.insert_slice %7 into %arg1[0, 0, 0, 0] [1, 1, 2, 4] [1, 1, 1, 1] : tensor<1x2x4xf32> into tensor<1x2x2x4xf32>
      "dialect.op"(%arg1) : (tensor<1x2x2x4xf32>) -> ()
      scf.yield %8 : tensor<1x2x2x4xf32>
    }
    scf.yield %1 : tensor<1x2x2x4xf32>
  }
  return %0 : tensor<1x2x2x4xf32>
}

//   CHECK-LABEL: func.func @dont_hoist_non_extract_insert_slice_usage_of_loop_carried_value
//     CHECK-NOT:   tensor.extract_slice
// CHECK-COUNT-2:   scf.for
// CHECK-COUNT-3:     tensor.extract_slice
//         CHECK:     tensor.insert_slice
// CHECK-COUNT-2:   scf.yield
//     CHECK-NOT:   tensor.insert_slice

// -----

func.func @dont_hoist_loop_dependent_slice_parameters(
    %input: tensor<1x9x9x3xf32>, %filter: tensor<3x3x3x16xf32>, %init: tensor<1x2x2x4xf32>,
    %offset0: index, %offset1: index, %offset2: index) -> tensor<1x2x2x4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %0 = scf.for %iv0 = %c0 to %c3 step %c1 iter_args(%arg0 = %init) -> (tensor<1x2x2x4xf32>) {
    %1 = scf.for %iv1 = %c0 to %c3 step %c1 iter_args(%arg1 = %arg0) -> (tensor<1x2x2x4xf32>) {
      %2 = affine.apply affine_map<(d0, d1) -> (d0 * 2 + d1)>(%offset0, %iv0)
      %3 = affine.apply affine_map<(d0, d1) -> (d0 * 2 + d1)>(%offset1, %iv1)
      %4 = tensor.extract_slice %input[%c0, %2, %3, %c0] [1, 1, 3, 3] [1, 1, 1, 1] : tensor<1x9x9x3xf32> to tensor<1x3x3xf32>
      %5 = tensor.extract_slice %filter[%iv0, %iv1, %c0, %offset2] [1, 1, 3, 4] [1, 1, 1, 1] : tensor<3x3x3x16xf32> to tensor<1x3x4xf32>
      %mod = affine.apply affine_map<(d0) -> (d0 mod 2)>(%iv1)
      %6 = tensor.extract_slice %arg1[%c0, %mod, %c0, %c0] [1, 1, 2, 4] [1, 1, 1, 1] : tensor<1x2x2x4xf32> to tensor<1x2x4xf32>
      %7 = linalg.conv_1d_nwc_wcf {dilations = dense<1> : vector<1xi64>, strides = dense<2> : vector<1xi64>}
            ins(%4, %5 : tensor<1x3x3xf32>, tensor<1x3x4xf32>) outs(%6 : tensor<1x2x4xf32>) -> tensor<1x2x4xf32>
      %8 = tensor.insert_slice %7 into %arg1[0, %mod, 0, 0] [1, 1, 2, 4] [1, 1, 1, 1] : tensor<1x2x4xf32> into tensor<1x2x2x4xf32>
      "dialect.op"(%arg1) : (tensor<1x2x2x4xf32>) -> ()
      scf.yield %8 : tensor<1x2x2x4xf32>
    }
    scf.yield %1 : tensor<1x2x2x4xf32>
  }
  return %0 : tensor<1x2x2x4xf32>
}

//   CHECK-LABEL: func.func @dont_hoist_loop_dependent_slice_parameters
//     CHECK-NOT:   tensor.extract_slice
// CHECK-COUNT-2:   scf.for
// CHECK-COUNT-3:     tensor.extract_slice
//         CHECK:     tensor.insert_slice
// CHECK-COUNT-2:   scf.yield
//     CHECK-NOT:   tensor.insert_slice

// -----

func.func @dont_hoist_slices_not_disjoint(
    %input: tensor<1x9x9x3xf32>, %filter: tensor<3x3x3x16xf32>, %init: tensor<1x2x2x4xf32>,
    %offset0: index, %offset1: index, %offset2: index) -> tensor<1x2x2x4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %0 = scf.for %iv0 = %c0 to %c3 step %c1 iter_args(%arg0 = %init) -> (tensor<1x2x2x4xf32>) {
    %1 = scf.for %iv1 = %c0 to %c3 step %c1 iter_args(%arg1 = %arg0) -> (tensor<1x2x2x4xf32>) {
      %2 = affine.apply affine_map<(d0, d1) -> (d0 * 2 + d1)>(%offset0, %iv0)
      %3 = affine.apply affine_map<(d0, d1) -> (d0 * 2 + d1)>(%offset1, %iv1)
      %4 = tensor.extract_slice %input[%c0, %2, %3, %c0] [1, 1, 3, 3] [1, 1, 1, 1] : tensor<1x9x9x3xf32> to tensor<1x3x3xf32>
      %5 = tensor.extract_slice %filter[%iv0, %iv1, %c0, %offset2] [1, 1, 3, 4] [1, 1, 1, 1] : tensor<3x3x3x16xf32> to tensor<1x3x4xf32>
      %6 = tensor.extract_slice %arg1[%c0, %c0, %c0, %c0] [1, 1, 2, 4] [1, 1, 1, 1] : tensor<1x2x2x4xf32> to tensor<1x2x4xf32>
      %7 = linalg.conv_1d_nwc_wcf {dilations = dense<1> : vector<1xi64>, strides = dense<2> : vector<1xi64>}
            ins(%4, %5 : tensor<1x3x3xf32>, tensor<1x3x4xf32>) outs(%6 : tensor<1x2x4xf32>) -> tensor<1x2x4xf32>
      %8 = tensor.insert_slice %7 into %arg1[0, 0, 0, 0] [1, 1, 2, 4] [1, 1, 1, 1] : tensor<1x2x4xf32> into tensor<1x2x2x4xf32>
      %9 = affine.apply affine_map<(d0, d1) -> (d0 * 2 + d1 + 2)>(%offset0, %iv0)
      %10 = affine.apply affine_map<(d0, d1) -> (d0 * 2 + d1)>(%offset1, %iv1)
      %11 = tensor.extract_slice %input[%c0, %9, %10, %c0] [1, 1, 3, 3] [1, 1, 1, 1] : tensor<1x9x9x3xf32> to tensor<1x3x3xf32>
      %12 = tensor.extract_slice %filter[%iv0, %iv1, %c0, %offset2] [1, 1, 3, 4] [1, 1, 1, 1] : tensor<3x3x3x16xf32> to tensor<1x3x4xf32>
      %13 = tensor.extract_slice %arg1[%c0, %c0, %c1, %c0] [1, 1, 2, 4] [1, 1, 1, 1] : tensor<1x2x2x4xf32> to tensor<1x2x4xf32>
      %14 = linalg.conv_1d_nwc_wcf {dilations = dense<1> : vector<1xi64>, strides = dense<2> : vector<1xi64>}
            ins(%11, %12 : tensor<1x3x3xf32>, tensor<1x3x4xf32>) outs(%13 : tensor<1x2x4xf32>) -> tensor<1x2x4xf32>
      %15 = tensor.insert_slice %14 into %8[0, 0, 1, 0] [1, 1, 2, 4] [1, 1, 1, 1] : tensor<1x2x4xf32> into tensor<1x2x2x4xf32>
      scf.yield %15 : tensor<1x2x2x4xf32>
    }
    scf.yield %1 : tensor<1x2x2x4xf32>
  }
  return %0 : tensor<1x2x2x4xf32>
}

//   CHECK-LABEL: func.func @dont_hoist_slices_not_disjoint
//     CHECK-NOT:   tensor.extract_slice
// CHECK-COUNT-2:   scf.for
// CHECK-COUNT-3:     tensor.extract_slice
//         CHECK:     tensor.insert_slice
// CHECK-COUNT-3:     tensor.extract_slice
//         CHECK:     tensor.insert_slice
// CHECK-COUNT-2:   scf.yield
//     CHECK-NOT:   tensor.insert_slice

// -----

func.func @dont_hoist_mismatched_extract_insert_slice(
    %input: tensor<1x9x9x3xf32>, %filter: tensor<3x3x3x16xf32>, %init: tensor<1x2x2x4xf32>,
    %offset0: index, %offset1: index, %offset2: index) -> tensor<1x2x2x4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %0 = scf.for %iv0 = %c0 to %c3 step %c1 iter_args(%arg0 = %init) -> (tensor<1x2x2x4xf32>) {
    %1 = scf.for %iv1 = %c0 to %c3 step %c1 iter_args(%arg1 = %arg0) -> (tensor<1x2x2x4xf32>) {
      %2 = affine.apply affine_map<(d0, d1) -> (d0 * 2 + d1)>(%offset0, %iv0)
      %3 = affine.apply affine_map<(d0, d1) -> (d0 * 2 + d1)>(%offset1, %iv1)
      %4 = tensor.extract_slice %input[%c0, %2, %3, %c0] [1, 1, 3, 3] [1, 1, 1, 1] : tensor<1x9x9x3xf32> to tensor<1x3x3xf32>
      %5 = tensor.extract_slice %filter[%iv0, %iv1, %c0, %offset2] [1, 1, 3, 4] [1, 1, 1, 1] : tensor<3x3x3x16xf32> to tensor<1x3x4xf32>
      %6 = tensor.extract_slice %arg1[%c0, %c0, %c0, %c0] [1, 1, 2, 4] [1, 1, 1, 1] : tensor<1x2x2x4xf32> to tensor<1x2x4xf32>
      %7 = linalg.conv_1d_nwc_wcf {dilations = dense<1> : vector<1xi64>, strides = dense<2> : vector<1xi64>}
            ins(%4, %5 : tensor<1x3x3xf32>, tensor<1x3x4xf32>) outs(%6 : tensor<1x2x4xf32>) -> tensor<1x2x4xf32>
      %8 = tensor.insert_slice %7 into %arg1[0, 1, 0, 0] [1, 1, 2, 4] [1, 1, 1, 1] : tensor<1x2x4xf32> into tensor<1x2x2x4xf32>
      scf.yield %8 : tensor<1x2x2x4xf32>
    }
    scf.yield %1 : tensor<1x2x2x4xf32>
  }
  return %0 : tensor<1x2x2x4xf32>
}

//   CHECK-LABEL: func.func @dont_hoist_mismatched_extract_insert_slice
//     CHECK-NOT:   tensor.extract_slice
// CHECK-COUNT-2:   scf.for
// CHECK-COUNT-3:     tensor.extract_slice
//         CHECK:     tensor.insert_slice
// CHECK-COUNT-2:   scf.yield
//     CHECK-NOT:   tensor.insert_slice
