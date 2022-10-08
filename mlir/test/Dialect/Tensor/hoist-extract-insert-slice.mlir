// RUN: mlir-opt -split-input-file -test-tensor-transform-patterns=test-hoist-extract-insert-slice -allow-unregistered-dialect -canonicalize %s | FileCheck %s

func.func @hoist_slices_in_double_loop(
    %input: tensor<1x9x9x3xf32>, %filter: tensor<3x3x3x16xf32>, %init: tensor<1x2x2x4xf32>,
    %offset0: index, %offset1: index, %offset2: index) -> tensor<1x2x2x4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %0 = scf.for %iv0 = %c0 to %c3 step %c1 iter_args(%arg0 = %init) -> (tensor<1x2x2x4xf32>) {
    %1 = scf.for %iv1 = %c0 to %c3 step %c1 iter_args(%arg1 = %arg0) -> (tensor<1x2x2x4xf32>) {
      %6 = tensor.extract_slice %arg1[%c0, %c0, %c0, %c0] [1, 1, 2, 4] [1, 1, 1, 1] : tensor<1x2x2x4xf32> to tensor<1x2x4xf32>
      %7 = "normal.compute"(%6) : (tensor<1x2x4xf32>) -> tensor<1x2x4xf32>
      %8 = tensor.insert_slice %7 into %arg1[0, 0, 0, 0] [1, 1, 2, 4] [1, 1, 1, 1] : tensor<1x2x4xf32> into tensor<1x2x2x4xf32>
      %13 = tensor.extract_slice %arg1[%c0, %c1, %c0, %c0] [1, 1, 2, 4] [1, 1, 1, 1] : tensor<1x2x2x4xf32> to tensor<1x2x4xf32>
      %14 = "normal.compute"(%13) : (tensor<1x2x4xf32>) -> tensor<1x2x4xf32>
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
//       CHECK:       %[[COMP0:.+]] = "normal.compute"(%[[FOR1_ARG0]])
//       CHECK:       %[[COMP1:.+]] = "normal.compute"(%[[FOR1_ARG1]])
//       CHECK:       scf.yield %[[COMP1]], %[[COMP0]] : tensor<1x2x4xf32>, tensor<1x2x4xf32>
//       CHECK:     }
//       CHECK:     scf.yield %[[FOR1]]#0, %[[FOR1]]#1 : tensor<1x2x4xf32>, tensor<1x2x4xf32>
//       CHECK:   }
//       CHECK:   %[[INSERT0:.+]] = tensor.insert_slice %[[FOR0]]#1 into %[[INIT]][0, 0, 0, 0] [1, 1, 2, 4] [1, 1, 1, 1]
//       CHECK:   %[[INSERT1:.+]] = tensor.insert_slice %[[FOR0]]#0 into %[[INSERT0]][0, 1, 0, 0] [1, 1, 2, 4] [1, 1, 1, 1]
//       CHECK:   return %[[INSERT1]] : tensor<1x2x2x4xf32>

// -----

func.func @hoist_long_extract_insert_chain(
    %input: tensor<1x9x9x3xf32>, %filter: tensor<3x3x3x16xf32>, %init: tensor<1x4x2x4xf32>,
    %offset0: index, %offset1: index, %offset2: index) -> tensor<1x4x2x4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %0 = scf.for %iv0 = %c0 to %c3 step %c1 iter_args(%arg0 = %init) -> (tensor<1x4x2x4xf32>) {
    %1 = scf.for %iv1 = %c0 to %c3 step %c1 iter_args(%arg1 = %arg0) -> (tensor<1x4x2x4xf32>) {
      %6 = tensor.extract_slice %arg1[%c0, %c0, %c0, %c0] [1, 1, 2, 4] [1, 1, 1, 1] : tensor<1x4x2x4xf32> to tensor<1x2x4xf32>
      %7 = "normal.compute"(%6) : (tensor<1x2x4xf32>) -> tensor<1x2x4xf32>
      %8 = tensor.insert_slice %7 into %arg1[0, 0, 0, 0] [1, 1, 2, 4] [1, 1, 1, 1] : tensor<1x2x4xf32> into tensor<1x4x2x4xf32>

      %13 = tensor.extract_slice %arg1[%c0, %c1, %c0, %c0] [1, 1, 2, 4] [1, 1, 1, 1] : tensor<1x4x2x4xf32> to tensor<1x2x4xf32>
      %14 = "normal.compute"(%13) : (tensor<1x2x4xf32>) -> tensor<1x2x4xf32>
      %15 = tensor.insert_slice %14 into %8[0, 1, 0, 0] [1, 1, 2, 4] [1, 1, 1, 1] : tensor<1x2x4xf32> into tensor<1x4x2x4xf32>

      %16 = tensor.extract_slice %arg1[%c0, %c2, %c0, %c0] [1, 1, 2, 4] [1, 1, 1, 1] : tensor<1x4x2x4xf32> to tensor<1x2x4xf32>
      %17 = "normal.compute"(%16) : (tensor<1x2x4xf32>) -> tensor<1x2x4xf32>
      %18 = tensor.insert_slice %17 into %15[0, 2, 0, 0] [1, 1, 2, 4] [1, 1, 1, 1] : tensor<1x2x4xf32> into tensor<1x4x2x4xf32>

      %19 = tensor.extract_slice %arg1[%c0, %c3, %c0, %c0] [1, 1, 2, 4] [1, 1, 1, 1] : tensor<1x4x2x4xf32> to tensor<1x2x4xf32>
      %20 = "normal.compute"(%19) : (tensor<1x2x4xf32>) -> tensor<1x2x4xf32>
      %21 = tensor.insert_slice %20 into %18[0, 3, 0, 0] [1, 1, 2, 4] [1, 1, 1, 1] : tensor<1x2x4xf32> into tensor<1x4x2x4xf32>
      scf.yield %21 : tensor<1x4x2x4xf32>
    }
    scf.yield %1 : tensor<1x4x2x4xf32>
  }
  return %0 : tensor<1x4x2x4xf32>
}

//   CHECK-LABEL: func.func @hoist_long_extract_insert_chain
// CHECK-COUNT-4:   tensor.extract_slice
// CHECK-COUNT-2:   scf.for
// CHECK-COUNT-2:   scf.yield
// CHECK-COUNT-4:   tensor.insert_slice

// -----

func.func @dont_hoist_non_extract_insert_slice_usage_of_loop_carried_value(
    %init: tensor<1x2x2x4xf32>,
    %offset0: index, %offset1: index, %offset2: index) -> tensor<1x2x2x4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %0 = scf.for %iv0 = %c0 to %c3 step %c1 iter_args(%arg0 = %init) -> (tensor<1x2x2x4xf32>) {
    %1 = scf.for %iv1 = %c0 to %c3 step %c1 iter_args(%arg1 = %arg0) -> (tensor<1x2x2x4xf32>) {
      %6 = tensor.extract_slice %arg1[%c0, %c0, %c0, %c0] [1, 1, 2, 4] [1, 1, 1, 1] : tensor<1x2x2x4xf32> to tensor<1x2x4xf32>
      %7 = "normal.compute"(%6) : (tensor<1x2x4xf32>) -> tensor<1x2x4xf32>
      %8 = tensor.insert_slice %7 into %arg1[0, 0, 0, 0] [1, 1, 2, 4] [1, 1, 1, 1] : tensor<1x2x4xf32> into tensor<1x2x2x4xf32>
      "blocking.usage"(%arg1) : (tensor<1x2x2x4xf32>) -> ()
      scf.yield %8 : tensor<1x2x2x4xf32>
    }
    scf.yield %1 : tensor<1x2x2x4xf32>
  }
  return %0 : tensor<1x2x2x4xf32>
}

//   CHECK-LABEL: func.func @dont_hoist_non_extract_insert_slice_usage_of_loop_carried_value
//     CHECK-NOT:   tensor.extract_slice
// CHECK-COUNT-2:   scf.for
//         CHECK:     tensor.extract_slice
//         CHECK:     tensor.insert_slice
// CHECK-COUNT-2:   scf.yield
//     CHECK-NOT:   tensor.insert_slice

// -----

func.func @dont_hoist_multi_insert_slice_uses(
    %init: tensor<1x2x2x4xf32>,
    %offset0: index, %offset1: index, %offset2: index) -> tensor<1x2x2x4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %0 = scf.for %iv0 = %c0 to %c3 step %c1 iter_args(%arg0 = %init) -> (tensor<1x2x2x4xf32>) {
    %1 = scf.for %iv1 = %c0 to %c3 step %c1 iter_args(%arg1 = %arg0) -> (tensor<1x2x2x4xf32>) {
      %6 = tensor.extract_slice %arg1[%c0, %c0, %c0, %c0] [1, 1, 2, 4] [1, 1, 1, 1] : tensor<1x2x2x4xf32> to tensor<1x2x4xf32>
      %7 = "normal.compute"(%6) : (tensor<1x2x4xf32>) -> tensor<1x2x4xf32>
      %8 = tensor.insert_slice %7 into %arg1[0, 0, 0, 0] [1, 1, 2, 4] [1, 1, 1, 1] : tensor<1x2x4xf32> into tensor<1x2x2x4xf32>
      "blocking.usage"(%8) : (tensor<1x2x2x4xf32>) -> ()
      scf.yield %8 : tensor<1x2x2x4xf32>
    }
    scf.yield %1 : tensor<1x2x2x4xf32>
  }
  return %0 : tensor<1x2x2x4xf32>
}

//   CHECK-LABEL: func.func @dont_hoist_multi_insert_slice_uses
//     CHECK-NOT:   tensor.extract_slice
// CHECK-COUNT-2:   scf.for
//         CHECK:     tensor.extract_slice
//         CHECK:     tensor.insert_slice
// CHECK-COUNT-2:   scf.yield
//     CHECK-NOT:   tensor.insert_slice

// -----

func.func @dont_hoist_loop_dependent_slice_parameters(
    %init: tensor<1x2x2x4xf32>,
    %offset0: index, %offset1: index, %offset2: index) -> tensor<1x2x2x4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %0 = scf.for %iv0 = %c0 to %c3 step %c1 iter_args(%arg0 = %init) -> (tensor<1x2x2x4xf32>) {
    %1 = scf.for %iv1 = %c0 to %c3 step %c1 iter_args(%arg1 = %arg0) -> (tensor<1x2x2x4xf32>) {
      %mod = affine.apply affine_map<(d0) -> (d0 mod 2)>(%iv1)
      %6 = tensor.extract_slice %arg1[%c0, %mod, %c0, %c0] [1, 1, 2, 4] [1, 1, 1, 1] : tensor<1x2x2x4xf32> to tensor<1x2x4xf32>
      %7 = "normal.compute"(%6) : (tensor<1x2x4xf32>) -> tensor<1x2x4xf32>
      %8 = tensor.insert_slice %7 into %arg1[0, %mod, 0, 0] [1, 1, 2, 4] [1, 1, 1, 1] : tensor<1x2x4xf32> into tensor<1x2x2x4xf32>
      scf.yield %8 : tensor<1x2x2x4xf32>
    }
    scf.yield %1 : tensor<1x2x2x4xf32>
  }
  return %0 : tensor<1x2x2x4xf32>
}

//   CHECK-LABEL: func.func @dont_hoist_loop_dependent_slice_parameters
//     CHECK-NOT:   tensor.extract_slice
// CHECK-COUNT-2:   scf.for
//         CHECK:     tensor.extract_slice
//         CHECK:     tensor.insert_slice
// CHECK-COUNT-2:   scf.yield
//     CHECK-NOT:   tensor.insert_slice

// -----

func.func @dont_hoist_out_of_dependent_loops(
    %init: tensor<1x2x2x4xf32>,
    %offset0: index, %offset1: index, %offset2: index) -> tensor<1x2x2x4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %output = scf.for %iv = %c0 to %c2 step %c1 iter_args(%arg = %init) -> (tensor<1x2x2x4xf32>) {
    %0 = scf.for %iv0 = %c0 to %c3 step %c1 iter_args(%arg0 = %arg) -> (tensor<1x2x2x4xf32>) {
      %1 = scf.for %iv1 = %c0 to %c3 step %c1 iter_args(%arg1 = %arg0) -> (tensor<1x2x2x4xf32>) {
        %6 = tensor.extract_slice %arg1[%c0, %iv, %c0, %c0] [1, 1, 2, 4] [1, 1, 1, 1] : tensor<1x2x2x4xf32> to tensor<1x2x4xf32>
        %7 = "normal.compute"(%6) : (tensor<1x2x4xf32>) -> tensor<1x2x4xf32>
        %8 = tensor.insert_slice %7 into %arg1[0, %iv, 0, 0] [1, 1, 2, 4] [1, 1, 1, 1] : tensor<1x2x4xf32> into tensor<1x2x2x4xf32>
        scf.yield %8 : tensor<1x2x2x4xf32>
      }
      scf.yield %1 : tensor<1x2x2x4xf32>
    }
    scf.yield %0 : tensor<1x2x2x4xf32>
  }
  return %output : tensor<1x2x2x4xf32>
}

//   CHECK-LABEL: func.func @dont_hoist_out_of_dependent_loops
//         CHECK:   scf.for
//         CHECK:     tensor.extract_slice
// CHECK-COUNT-2:     scf.for
// CHECK-COUNT-2:     scf.yield
//         CHECK:     tensor.insert_slice
//         CHECK:   scf.yield

// -----

func.func @dont_hoist_insert_slices_not_disjoint(
    %init: tensor<1x2x2x4xf32>,
    %offset0: index, %offset1: index, %offset2: index) -> tensor<1x2x2x4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %0 = scf.for %iv0 = %c0 to %c3 step %c1 iter_args(%arg0 = %init) -> (tensor<1x2x2x4xf32>) {
    %1 = scf.for %iv1 = %c0 to %c3 step %c1 iter_args(%arg1 = %arg0) -> (tensor<1x2x2x4xf32>) {
      %6 = tensor.extract_slice %arg1[%c0, %c0, %c0, %c0] [1, 1, 2, 4] [1, 1, 1, 1] : tensor<1x2x2x4xf32> to tensor<1x2x4xf32>
      %7 = "normal.compute"(%6) : (tensor<1x2x4xf32>) -> tensor<1x2x4xf32>
      %8 = tensor.insert_slice %7 into %arg1[0, 0, 0, 0] [1, 1, 2, 4] [1, 1, 1, 1] : tensor<1x2x4xf32> into tensor<1x2x2x4xf32>
      %13 = tensor.extract_slice %arg1[%c0, %c0, %c1, %c0] [1, 1, 2, 4] [1, 1, 1, 1] : tensor<1x2x2x4xf32> to tensor<1x2x4xf32>
      %14 = "normal.compute"(%13) : (tensor<1x2x4xf32>) -> tensor<1x2x4xf32>
      %15 = tensor.insert_slice %14 into %8[0, 0, 1, 0] [1, 1, 2, 4] [1, 1, 1, 1] : tensor<1x2x4xf32> into tensor<1x2x2x4xf32>
      scf.yield %15 : tensor<1x2x2x4xf32>
    }
    scf.yield %1 : tensor<1x2x2x4xf32>
  }
  return %0 : tensor<1x2x2x4xf32>
}

//   CHECK-LABEL: func.func @dont_hoist_insert_slices_not_disjoint
//     CHECK-NOT:   tensor.extract_slice
// CHECK-COUNT-2:   scf.for
//         CHECK:     tensor.extract_slice
//         CHECK:     tensor.insert_slice
//         CHECK:     tensor.extract_slice
//         CHECK:     tensor.insert_slice
// CHECK-COUNT-2:   scf.yield
//     CHECK-NOT:   tensor.insert_slice

// -----

func.func @dont_hoist_mismatched_extract_insert_slice(
    %init: tensor<1x2x2x4xf32>,
    %offset0: index, %offset1: index, %offset2: index) -> tensor<1x2x2x4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %0 = scf.for %iv0 = %c0 to %c3 step %c1 iter_args(%arg0 = %init) -> (tensor<1x2x2x4xf32>) {
    %1 = scf.for %iv1 = %c0 to %c3 step %c1 iter_args(%arg1 = %arg0) -> (tensor<1x2x2x4xf32>) {
      %6 = tensor.extract_slice %arg1[%c0, %c0, %c0, %c0] [1, 1, 2, 4] [1, 1, 1, 1] : tensor<1x2x2x4xf32> to tensor<1x2x4xf32>
      %7 = "normal.compute"(%6) : (tensor<1x2x4xf32>) -> tensor<1x2x4xf32>
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
//         CHECK:     tensor.extract_slice
//         CHECK:     tensor.insert_slice
// CHECK-COUNT-2:   scf.yield
//     CHECK-NOT:   tensor.insert_slice
