// RUN: mlir-opt %s -allow-unregistered-dialect -test-decompose-affine-ops -split-input-file | FileCheck %s

// CHECK-DAG: #[[$div32div4timesm4:.*]] = affine_map<()[s0] -> (((s0 floordiv 32) floordiv 4) * -4)>
// CHECK-DAG: #[[$add:.*]] = affine_map<()[s0, s1] -> (s0 + s1)>
// CHECK-DAG: #[[$div32:.*]] = affine_map<()[s0] -> (s0 floordiv 32)>

// CHECK-LABEL:  func.func @simple_test
//  CHECK-SAME:  %[[I0:[0-9a-zA-Z]+]]: index,
//  CHECK-SAME:  %[[I1:[0-9a-zA-Z]+]]: index,
//  CHECK-SAME:  %[[I2:[0-9a-zA-Z]+]]: index,
//  CHECK-SAME:  %[[LB:[0-9a-zA-Z]+]]: index,
//  CHECK-SAME:  %[[UB:[0-9a-zA-Z]+]]: index,
//  CHECK-SAME:  %[[STEP:[0-9a-zA-Z]+]]: index
func.func @simple_test(%0: index, %1: index, %2: index, %lb: index, %ub: index, %step: index) {
  scf.for %i = %lb to %ub step %step {
    scf.for %j = %lb to %ub step %step {
      %a = affine.apply affine_map<(d0)[s0] -> ((d0 floordiv 32) mod 4 + s0 + 42)>(%j)[%i]
      "some_side_effecting_consumer"(%a) : (index) -> ()
    }
  }
  return
}

// // -----

// // CHECK-DAG: #[[$div4:.*]] = affine_map<()[s0] -> (s0 floordiv 4)>
// // CHECK-DAG: #[[$times32:.*]] = affine_map<()[s0] -> (s0 * 32)>
// // CHECK-DAG: #[[$times16:.*]] = affine_map<()[s0] -> (s0 * 16)>
// // CHECK-DAG: #[[$add:.*]] = affine_map<()[s0, s1] -> (s0 + s1)>
// // CHECK-DAG: #[[$div4timesm32:.*]] = affine_map<()[s0] -> ((s0 floordiv 4) * -32)>
// // CHECK-DAG: #[[$times8:.*]] = affine_map<()[s0] -> (s0 * 8)>
// // CHECK-DAG: #[[$div32div4timesm4:.*]] = affine_map<()[s0] -> (((s0 floordiv 32) floordiv 4) * -4)>
// // CHECK-DAG: #[[$div32:.*]] = affine_map<()[s0] -> (s0 floordiv 32)>

// // CHECK-LABEL:  func.func @test
// //  CHECK-SAME:  %[[I0:[0-9a-zA-Z]+]]: index,
// //  CHECK-SAME:  %[[I1:[0-9a-zA-Z]+]]: index,
// //  CHECK-SAME:  %[[I2:[0-9a-zA-Z]+]]: index,
// //  CHECK-SAME:  %[[LB:[0-9a-zA-Z]+]]: index,
// //  CHECK-SAME:  %[[UB:[0-9a-zA-Z]+]]: index,
// //  CHECK-SAME:  %[[STEP:[0-9a-zA-Z]+]]: index
// func.func @test(%0: index, %1: index, %2: index, %lb: index, %ub: index, %step: index) {
//     %c2 = arith.constant 2 : index
//     %c6 = arith.constant 6 : index

//     // CHECK: %[[R0:.*]] = affine.apply #[[$div4]]()[%[[I0]]]
//     // CHECK: %[[R1:.*]] = affine.apply #[[$times32]]()[%[[I2]]]
//     // CHECK: %[[R2:.*]] = affine.apply #[[$times16]]()[%[[I1]]]
//     // CHECK: %[[R3:.*]] = affine.apply #[[$add]]()[%[[R0]], %[[R1]]]

//     // I1 * 16 + I2 * 32 + I0 floordiv 4
//     // CHECK: %[[b:.*]] = affine.apply #[[$add]]()[%[[R3]], %[[R2]]]

//     // (I0 floordiv 4) * 32
//     // CHECK: %[[R5:.*]] = affine.apply #[[$div4timesm32]]()[%[[I0]]]
//     // 8 * I0
//     // CHECK: %[[R6:.*]] = affine.apply #[[$times8]]()[%[[I0]]]
//     // 8 * I0 + (I0 floordiv 4) * 32
//     // CHECK: %[[c:.*]] = affine.apply #[[$add]]()[%[[R5]], %[[R6]]]

//     // CHECK: scf.for %[[i:.*]] =
//     scf.for %i = %lb to %ub step %step {
//       // remainder from %a not hoisted above %i.
//       // CHECK: %[[R8:.*]] = affine.apply #[[$times32]]()[%[[i]]]
//       // CHECK: %[[a:.*]] = affine.apply #[[$add]]()[%[[b]], %[[R8]]]

//       // CHECK: scf.for %[[j:.*]] =
//       scf.for %j = %lb to %ub step %step {
//         // Gets hoisted partially to i and rest outermost.
//         // The hoisted part is %b.
//         %a = affine.apply affine_map<()[s0, s1, s2, s3] -> (s1 * 16 + s2 * 32 + s3 * 32 + s0 floordiv 4)>()[%0, %1, %2, %i]

//         // Gets completely hoisted 
//         %b = affine.apply affine_map<()[s0, s1, s2] -> (s1 * 16 + s2 * 32 + s0 floordiv 4)>()[%0, %1, %2]

//         // Gets completely hoisted 
//         %c = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 4) * 32)>()[%0]
 
//         // 32 * %j + %c remains here, the rest is hoisted.
//         // CHECK-DAG: %[[R10:.*]] = affine.apply #[[$times32]]()[%[[j]]]
//         // CHECK-DAG: %[[d:.*]] = affine.apply #[[$add]]()[%[[c]], %[[R10]]]
//         %d = affine.apply affine_map<()[s0, s1] -> (s0 * 8 + s1 * 32 - (s0 floordiv 4) * 32)>()[%0, %j]

//         // CHECK: scf.for %[[k:.*]] =
//         scf.for %k = %lb to %ub step %step {
//           // CHECK: %[[R12:.*]] = affine.apply #[[$div32div4timesm4]]()[%[[k]]]

//           // CHECK: %[[R13:.*]] = affine.apply #[[$div32]]()[%[[k]]]
//           %e = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 8 - (s1 floordiv 4) * 32)>()[%k, %0]

//           // CHECK: %[[f:.*]] = affine.apply #[[$add]]()[%[[R12]], %[[R13]]]
//           %f = affine.apply affine_map<(d0)[s0] -> ((d0 floordiv 32) mod 4 + s0)>(%k)[%j]
//           %g = affine.apply affine_map<()[s0, s1, s2, s3] -> (s0 + s2 * 16 + s3 * 32 + s1 floordiv 4)>()[%k, %0, %1, %2]
          
//           // "some_side_effecting_consumer"(%[[a]]) : (index) -> ()
//           "some_side_effecting_consumer"(%a) : (index) -> ()
//           // "some_side_effecting_consumer"(%[[b]]) : (index) -> ()
//           "some_side_effecting_consumer"(%b) : (index) -> ()
//           // "some_side_effecting_consumer"(%[[c]]) : (index) -> ()
//           "some_side_effecting_consumer"(%c) : (index) -> ()
//           // "some_side_effecting_consumer"(%[[d]]) : (index) -> ()
//           "some_side_effecting_consumer"(%d) : (index) -> ()
//           // "some_side_effecting_consumer"(%[[R7]]) : (index) -> ()
//           "some_side_effecting_consumer"(%e) : (index) -> ()
//           // "some_side_effecting_consumer"(%[[f]]) : (index) -> ()
//           "some_side_effecting_consumer"(%f) : (index) -> ()
//           // "some_side_effecting_consumer"(%[[R4]]) : (index) -> ()
//           "some_side_effecting_consumer"(%g) : (index) -> ()
//         }
//     }
//   }   
//   return
// }
