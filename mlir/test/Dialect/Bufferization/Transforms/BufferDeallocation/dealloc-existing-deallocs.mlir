// DEFINE: %{canonicalize} = -canonicalize=enable-patterns="bufferization-skip-extract-metadata-of-alloc,bufferization-erase-always-false-dealloc,bufferization-erase-empty-dealloc,bufferization-dealloc-remove-duplicate-retained-memrefs,bufferization-dealloc-remove-duplicate-dealloc-memrefs",region-simplify=false

// RUN: mlir-opt -verify-diagnostics -buffer-deallocation %{canonicalize} \
// RUN:  --buffer-deallocation-simplification %{canonicalize} -split-input-file %s | FileCheck %s
// RUN: mlir-opt -verify-diagnostics -buffer-deallocation=private-function-dynamic-ownership=true -split-input-file %s > /dev/null

// Ensure we free the realloc, not the alloc.

func.func @auto_dealloc() {
  %c10 = arith.constant 10 : index
  %c100 = arith.constant 100 : index
  %alloc = memref.alloc(%c10) : memref<?xi32>
  %realloc = memref.realloc %alloc(%c100) : memref<?xi32> to memref<?xi32>
  return
}

// CHECK-LABEL: func @auto_dealloc()
//   CHECK-DAG:   %[[C10:.*]] = arith.constant 10 : index
//   CHECK-DAG:   %[[C100:.*]] = arith.constant 100 : index
//       CHECK:  %[[A:.*]] = memref.alloc(%[[C10]])
//  iCHECK-NOT:  bufferization.dealloc
//       CHECK:  %[[R:.*]] = memref.realloc %alloc(%[[C100]])
//   CHECK-NOT:  bufferization.dealloc{{.*}}%[[A]]
//       CHECK:  bufferization.dealloc (%[[R]] :{{.*}}) if (%true{{[0-9_]*}})
//   CHECK-NOT:  bufferization.dealloc

// -----

func.func @auto_dealloc_inside_nested_region(%arg0: memref<?xi32>, %arg1: i1) -> memref<?xi32> {
  %c100 = arith.constant 100 : index
  %0 = scf.if %arg1 -> memref<?xi32> {
    %realloc = memref.realloc %arg0(%c100) : memref<?xi32> to memref<?xi32>
    scf.yield %realloc : memref<?xi32>
  } else {
    scf.yield %arg0 : memref<?xi32>
  }
  return %0 : memref<?xi32>
}

// CHECK-LABEL: func @auto_dealloc_inside_nested_region
//  CHECK-SAME: [[ARG0:%.+]]:{{.*}},
//  CHECK-SAME: [[ARG1:%.+]]:{{.*}})
//       CHECK: scf.if [[ARG1]]
//       CHECK:   [[REALLOC:%.+]] = memref.realloc [[ARG0]]
//       CHECK:   scf.yield [[REALLOC]], %true
//       CHECK:   scf.yield [[ARG0]], %false
