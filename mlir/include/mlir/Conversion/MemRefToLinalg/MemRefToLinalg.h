//===- MemRefToLinalg.h - MemRef to Linalg dialect conversion ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_MEMREFTOLLVM_MEMREFTOLINALG_H
#define MLIR_CONVERSION_MEMREFTOLLVM_MEMREFTOLINALG_H

#include "mlir/Pass/Pass.h"

namespace mlir {
class ModuleOp;

#define GEN_PASS_DECL_CONVERTMEMREFTOLINALG
#include "mlir/Conversion/Passes.h.inc"

/// Creates a pass to convert Memref ops to Linalg ops.
std::unique_ptr<OperationPass<ModuleOp>> createConvertMemRefToLinalgPass();

} // namespace mlir

#endif // MLIR_CONVERSION_MEMREFTOLLVM_MEMREFTOLINALG_H
