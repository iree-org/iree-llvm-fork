//===-- CUDNN.td - CUDNN backend API dialect operation *- tablegen -*------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the basic operations for the CUDNN dialect.
//
// This CUDNN provides a bridge between the target agnostic GPU and Vector
// dialects and lower level NVVM dialect. This allow representing PTX specific
// operations while using MLIR high level concepts like memref and 2-D vector.
//
// Ops semantic are going to be based on vendor specific PTX defintion:
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
//
//===----------------------------------------------------------------------===//

#ifndef CUDNN
#define CUDNN

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

def CUDNN_Dialect : Dialect {
  let name = "cudnn";
  let cppNamespace = "::mlir::cudnn";
  let description = [{
    The `CUDNN` dialect provides an export dialect to CUDNN backend API.
  }];

  let useDefaultTypePrinterParser = 1;
  let useFoldAPI = kEmitFoldAdaptorFolder;
}

#endif // CUDNN
