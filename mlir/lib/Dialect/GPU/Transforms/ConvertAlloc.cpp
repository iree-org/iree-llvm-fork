//===- ConvertAlloc.cpp - Convert alloc to gpu ones =//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements utilities to generate mappings for parallel loops to
// GPU devices.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/GPU/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineMap.h"

namespace mlir {
#define GEN_PASS_DEF_GPUTRANSFORMALLOCPASS
#include "mlir/Dialect/GPU/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::gpu;


static void copyToGPUAllocAndReplace(Value memref) {
  OpBuilder builder(memref.getContext());
  if(auto def = memref.getDefiningOp()) {
    builder.setInsertionPointAfter(def);
  } else {
      Block *block = memref.cast<BlockArgument>().getOwner();
      builder.setInsertionPointToStart(block);
  }

  auto newAlloc = builder.create<gpu::AllocOp>(
      memref.getLoc(), memref.getType(), nullptr, ArrayRef<Value>({}),
      ArrayRef<Value>({}), ArrayRef<Value>({}));
  memref.replaceAllUsesWith(newAlloc.getResult(0));
  Type empty;
  builder.create<gpu::MemcpyOp>(
      memref.getLoc(), empty, ArrayRef<Value>({}),
      newAlloc.getResult(0), memref);
}

static void replaceAlloc(memref::AllocOp alloc) {
  OpBuilder builder(alloc);
  auto newAlloc = builder.create<gpu::AllocOp>(
      alloc.getLoc(), alloc.getType(), nullptr, ArrayRef<Value>({}),
      alloc.getDynamicSizes(), ArrayRef<Value>({}));
  alloc.getResult().replaceAllUsesWith(newAlloc.getResult(0));
  // Replace dealloc uses.
  SmallVector<Operation*> deallocs;
  for(Operation* userOp : newAlloc->getUsers()) {
    if(isa<memref::DeallocOp>(userOp))
      deallocs.push_back(userOp);
  }
  for (Operation *dealloc : deallocs) {
    builder.setInsertionPoint(dealloc);
     builder.create<gpu::DeallocOp>(
        alloc.getLoc(),Type(), ArrayRef<Value>({}),
        newAlloc.getResult(0));
      dealloc->erase();
  }
  alloc->erase();

}

static void registerOnDevice(Value memref) {
  OpBuilder builder(memref.getContext());
  if(Operation* def = memref.getDefiningOp()) {
    builder.setInsertionPointAfter(def);
  } else {
    builder.setInsertionPointToStart(memref.getParentBlock());
  }
  Value unrankedMemRef = builder.create<memref::CastOp>(
      memref.getLoc(),
      UnrankedMemRefType::get(
          memref.getType().cast<MemRefType>().getElementType(), 0),
      memref);
  builder.create<gpu::HostRegisterOp>(memref.getLoc(), unrankedMemRef);
}

namespace {
struct GpuTransformAllocPass
    : public impl::GpuTransformAllocPassBase<GpuTransformAllocPass> {
  void runOnOperation() override {
    llvm::SetVector<Value> deviceMemref;
    getOperation()->walk([&deviceMemref](gpu::LaunchFuncOp launch) {
      for (Value operand : launch.getKernelOperands()) {
        if (operand.getType().isa<MemRefType>()) {
          Value memref = operand;
          while (auto defOp = memref.getDefiningOp()) {
            if(isa<memref::SubViewOp, memref::CollapseShapeOp>(defOp)) {
              memref = defOp->getOperand(0);
              continue;
            }
            break;
          }
          deviceMemref.insert(memref);
        }
      }
    });
    for(Value memref : deviceMemref) {
      auto alloc = memref.getDefiningOp<memref::AllocOp>();
      if (!alloc) {
        // registerOnDevice(memref);
        copyToGPUAllocAndReplace(memref);
        continue;
      }
      bool onlyUsedOnDevice = true;
      SmallVector<Operation*> users(alloc->getUsers().begin(),alloc->getUsers().end());
      while (!users.empty()) {
        Operation *userOp = users.back();
        users.pop_back();
        if (isa<memref::SubViewOp, memref::CollapseShapeOp>(userOp)) {
          users.append(userOp->getUsers().begin(), userOp->getUsers().end());
          continue;
        }
        if(!isa<gpu::LaunchFuncOp, memref::DeallocOp>(userOp)) {
          onlyUsedOnDevice = false;
          break;
        }
      }
      if(onlyUsedOnDevice)
        replaceAlloc(alloc);
      else
        registerOnDevice(memref);
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createGpuTransformAllocPass() {
  return std::make_unique<GpuTransformAllocPass>();
}
