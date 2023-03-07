//===- CUDNNDialect.cpp - CUDNN dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/CUDNN/IR/CUDNNDialect.h"
#include "mlir/Dialect/CUDNN/IR/CUDNNOps.h"
#include "mlir/Dialect/CUDNN/IR/CUDNNTypes.h"

using namespace mlir;
using namespace mlir::cudnn;

#include "mlir/Dialect/CUDNN/IR/CUDNNOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// CUDNN dialect.
//===----------------------------------------------------------------------===//

void CUDNNDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/CUDNN/IR/CUDNNOps.cpp.inc"
      >();
  registerTypes();
}
