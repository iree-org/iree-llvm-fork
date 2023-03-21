//===- CUDNNToLLVM.cpp - CUDNN to LLVM dialect conversion -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Converts functions fully consisting of cudnn ops to LLVM calls.
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/CUDNN/IR/CUDNNDialect.h"
#include "mlir/Dialect/CUDNN/IR/CUDNNOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"
#include <cudnn_backend.h>

using namespace mlir;
using namespace mlir::cudnn;

namespace {

// Builder of cudnn backend API calls.
struct BuildContext {
  BuildContext(ModuleOp module, ImplicitLocOpBuilder &builder)
      : b(builder), module(module) {
    context = b.getContext();
    i32Ty = b.getI32Type();
    i64Ty = b.getI64Type();
  }

  // Helper function to create a CUDNN_BACKEND_TENSOR_DESCRIPTOR for given
  // Value with TensorDescType result type.
  Value createDescriptorFor(Value val);

  // Create a new descriptor of given kind.
  Value createDescriptor(int32_t kind);

  // Sets a descriptor attribute with int64_t attributes.
  void setDescriptorAttribute(Value descriptor,
                              cudnnBackendAttributeName_t attributeName,
                              cudnnBackendAttributeType_t attributeType,
                              ArrayRef<int64_t> value);

  // Sets a descriptor attribute with double attributes.
  void setDescriptorAttribute(Value descriptor,
                              cudnnBackendAttributeName_t attributeName,
                              cudnnBackendAttributeType_t attributeType,
                              ArrayRef<double> value);

  // Sets a descriptor with already lowered value.
  void setDescriptorAttribute(Value descriptor,
                              cudnnBackendAttributeName_t attributeName,
                              cudnnBackendAttributeType_t attributeType,
                              Value lowered);

  // Finalize building the descriptor.
  void finalizeDescriptor(Value descriptor);

  // Destroy descriptor.
  void destroyDescriptor(Value descriptor);

  // Returns string that is mostly unique.
  // Note: This is not sufficient in general, just for quick testing.
  std::string getUnique(StringRef str) {
    return llvm::formatv("__const.{0}.{1}_{2}", graphName, str, ++ctr).str();
  }
  std::string getUnique(cudnnBackendAttributeName_t attributeName) {
    switch (attributeName) {
    case CUDNN_ATTR_TENSOR_DIMENSIONS:
      return getUnique("dims");
    case CUDNN_ATTR_TENSOR_STRIDES:
      return getUnique("strides");
    case CUDNN_ATTR_TENSOR_UNIQUE_ID:
      return getUnique("uid");
    default:
      return getUnique("attr");
    }
  }

  Type getVoidPtrType() {
    return LLVM::LLVMPointerType::get(IntegerType::get(context, 8));
  }

  Value i1(bool val) { return b.create<LLVM::ConstantOp>(b.getBoolAttr(val)); }
  Value i32(int32_t val) {
    return b.create<LLVM::ConstantOp>(b.getI32IntegerAttr(val));
  }
  Value i64(int64_t val) {
    return b.create<LLVM::ConstantOp>(b.getI64IntegerAttr(val));
  }

  // Helper functions that finds or inserts function reference.
  FlatSymbolRefAttr getOrInsertCreateDescriptorFn();
  FlatSymbolRefAttr getOrInsertDestroyDescriptorFn();
  FlatSymbolRefAttr getOrInsertFinalizeDescriptorFn();
  FlatSymbolRefAttr getOrInsertGetAttributeFn();
  FlatSymbolRefAttr getOrInsertSetAttributeFn();

  FlatSymbolRefAttr cudnnBackendCreateDescriptorRef = nullptr;
  FlatSymbolRefAttr cudnnBackendDestroyDescriptorRef = nullptr;
  FlatSymbolRefAttr cudnnBackendFinalizeDescriptorRef = nullptr;
  FlatSymbolRefAttr cudnnBackendGetAttributeRef = nullptr;
  FlatSymbolRefAttr cudnnBackendSetAttributeRef = nullptr;

  // For uniquing.
  std::string graphName;
  int64_t ctr = 0;

  // Cached types, mostly to reduce typing.
  Type i32Ty, i64Ty;

  llvm::MapVector<Value, Value> tensorToDescriptor;

  ImplicitLocOpBuilder &b;
  ModuleOp module;
  MLIRContext *context;
};

FlatSymbolRefAttr BuildContext::getOrInsertCreateDescriptorFn() {
  if (cudnnBackendCreateDescriptorRef)
    return cudnnBackendCreateDescriptorRef;
  if (module.lookupSymbol<LLVM::LLVMFuncOp>("cudnnBackendCreateDescriptor"))
    return cudnnBackendCreateDescriptorRef =
               SymbolRefAttr::get(context, "cudnnBackendCreateDescriptor");

  auto *context = module.getContext();
  // Create a function declaration with signature:
  //   (i32, !llvm.ptr<ptr<i8>>) -> i32
  auto llvmPtrPtrTy = LLVM::LLVMPointerType::get(getVoidPtrType());
  auto i32Ty = IntegerType::get(context, 32);
  auto llvmFnType = LLVM::LLVMFunctionType::get(i32Ty, {i32Ty, llvmPtrPtrTy},
                                                /*isVarArg=*/false);
  OpBuilder::InsertionGuard insertGuard(b);
  b.setInsertionPointToStart(module.getBody());
  b.create<LLVM::LLVMFuncOp>(module.getLoc(), "cudnnBackendCreateDescriptor",
                             llvmFnType);
  return cudnnBackendCreateDescriptorRef =
             SymbolRefAttr::get(context, "cudnnBackendCreateDescriptor");
}

FlatSymbolRefAttr BuildContext::getOrInsertDestroyDescriptorFn() {
  if (cudnnBackendDestroyDescriptorRef)
    return cudnnBackendDestroyDescriptorRef;
  if (module.lookupSymbol<LLVM::LLVMFuncOp>("cudnnBackendDestroyDescriptor"))
    return cudnnBackendDestroyDescriptorRef =
               SymbolRefAttr::get(context, "cudnnBackendDestroyDescriptor");

  auto *context = module.getContext();
  // Create a function declaration with signature:
  //   (!llvm.ptr<ptr<i8>>) -> i32
  auto llvmI8PtrTy = getVoidPtrType();
  auto i32Ty = IntegerType::get(context, 32);
  auto llvmFnType = LLVM::LLVMFunctionType::get(i32Ty, llvmI8PtrTy,
                                                /*isVarArg=*/false);
  OpBuilder::InsertionGuard insertGuard(b);
  b.setInsertionPointToStart(module.getBody());
  b.create<LLVM::LLVMFuncOp>(module.getLoc(), "cudnnBackendDestroyDescriptor",
                             llvmFnType);
  return cudnnBackendCreateDescriptorRef =
             SymbolRefAttr::get(context, "cudnnBackendDestroyDescriptor");
}

FlatSymbolRefAttr BuildContext::getOrInsertFinalizeDescriptorFn() {
  if (cudnnBackendFinalizeDescriptorRef)
    return cudnnBackendFinalizeDescriptorRef;
  if (module.lookupSymbol<LLVM::LLVMFuncOp>("cudnnBackendFinalizeDescriptor"))
    return cudnnBackendFinalizeDescriptorRef =
               SymbolRefAttr::get(context, "cudnnBackendFinalizeDescriptor");

  auto *context = module.getContext();
  // Create a function declaration with signature:
  //   (!llvm.ptr<ptr<i8>>) -> i32
  auto llvmI8PtrTy = getVoidPtrType();
  auto i32Ty = IntegerType::get(context, 32);
  auto llvmFnType = LLVM::LLVMFunctionType::get(i32Ty, llvmI8PtrTy,
                                                /*isVarArg=*/false);
  OpBuilder::InsertionGuard insertGuard(b);
  b.setInsertionPointToStart(module.getBody());
  b.create<LLVM::LLVMFuncOp>(module.getLoc(), "cudnnBackendFinalizeDescriptor",
                             llvmFnType);
  return cudnnBackendFinalizeDescriptorRef =
             SymbolRefAttr::get(context, "cudnnBackendFinalizeDescriptor");
}

FlatSymbolRefAttr BuildContext::getOrInsertGetAttributeFn() {
  if (cudnnBackendGetAttributeRef)
    return cudnnBackendGetAttributeRef;
  if (module.lookupSymbol<LLVM::LLVMFuncOp>("cudnnBackendGetAttribute"))
    return cudnnBackendGetAttributeRef =
               SymbolRefAttr::get(context, "cudnnBackendGetAttribute");

  auto *context = module.getContext();
  auto llvmI8PtrTy = getVoidPtrType();
  auto i64PtrTy = LLVM::LLVMPointerType::get(IntegerType::get(context, 64));
  auto llvmFnType = LLVM::LLVMFunctionType::get(
      i32Ty, {llvmI8PtrTy, i32Ty, i32Ty, i64Ty, i64PtrTy, llvmI8PtrTy},
      /*isVarArg=*/false);

  OpBuilder::InsertionGuard insertGuard(b);
  b.setInsertionPointToStart(module.getBody());
  b.create<LLVM::LLVMFuncOp>(module.getLoc(), "cudnnBackendGetAttribute",
                             llvmFnType);
  return cudnnBackendGetAttributeRef =
             SymbolRefAttr::get(context, "cudnnBackendGetAttribute");
}

FlatSymbolRefAttr BuildContext::getOrInsertSetAttributeFn() {
  if (cudnnBackendSetAttributeRef)
    return cudnnBackendSetAttributeRef;
  if (module.lookupSymbol<LLVM::LLVMFuncOp>("cudnnBackendSetAttribute"))
    return cudnnBackendSetAttributeRef =
               SymbolRefAttr::get(context, "cudnnBackendSetAttribute");

  auto *context = module.getContext();
  auto llvmI8PtrTy = getVoidPtrType();
  auto llvmFnType = LLVM::LLVMFunctionType::get(
      i32Ty, {llvmI8PtrTy, i32Ty, i32Ty, i64Ty, llvmI8PtrTy},
      /*isVarArg=*/false);

  OpBuilder::InsertionGuard insertGuard(b);
  b.setInsertionPointToStart(module.getBody());
  b.create<LLVM::LLVMFuncOp>(module.getLoc(), "cudnnBackendSetAttribute",
                             llvmFnType);
  return cudnnBackendSetAttributeRef =
             SymbolRefAttr::get(context, "cudnnBackendSetAttribute");
}

void BuildContext::setDescriptorAttribute(
    Value descriptor, cudnnBackendAttributeName_t attributeName,
    cudnnBackendAttributeType_t attributeType, ArrayRef<int64_t> value) {
  LLVM::GlobalOp global;
  // Create a global constant.
  {
    OpBuilder::InsertionGuard insertGuard(b);
    b.setInsertionPointToStart(module.getBody());
    auto type = LLVM::LLVMArrayType::get(i64Ty, value.size());
    global = b.create<LLVM::GlobalOp>(
        type, /*isConstant=*/true, LLVM::Linkage::Private,
        getUnique(attributeName), b.getI64TensorAttr(value),
        /*alignment=*/16, /*addrSpace=*/0,
        /*dsoLocal=*/true);
    // TODO: unnamed_addr
  }

  auto globalPtr = b.create<LLVM::BitcastOp>(
      getVoidPtrType(), b.create<LLVM::AddressOfOp>(global));
  auto toTy = LLVM::LLVMPointerType::get(LLVM::LLVMArrayType::get(i64Ty, 4));
  auto alloca = b.create<LLVM::BitcastOp>(
      getVoidPtrType(), b.create<LLVM::AllocaOp>(toTy, i32(1), 16));
  b.create<LLVM::MemcpyOp>(alloca, globalPtr, /*size=*/i64(8 * value.size()),
                           /*isVolatile=*/i1(false));

  auto nameVal = i32(attributeName);
  auto typeVal = i32(attributeType);
  auto valSize = i64(value.size());
  b.create<LLVM::CallOp>(
      ArrayRef<Type>{i32Ty}, getOrInsertSetAttributeFn(),
      ArrayRef<Value>{descriptor, nameVal, typeVal, valSize, alloca});
}

void BuildContext::setDescriptorAttribute(
    Value descriptor, cudnnBackendAttributeName_t attributeName,
    cudnnBackendAttributeType_t attributeType, ArrayRef<double> value) {
  LLVM::GlobalOp global;
  // Create a global constant.
  {
    OpBuilder::InsertionGuard insertGuard(b);
    b.setInsertionPointToStart(module.getBody());
    auto type = LLVM::LLVMArrayType::get(i64Ty, value.size());
    global = b.create<LLVM::GlobalOp>(
        type, /*isConstant=*/true, LLVM::Linkage::Private,
        getUnique(attributeName),
        DenseElementsAttr::get(
            RankedTensorType::get(static_cast<int64_t>(value.size()),
                                  b.getF64Type()),
            value),
        /*alignment=*/16, /*addrSpace=*/0,
        /*dsoLocal=*/true);
    // TODO: unnamed_addr
  }

  auto globalPtr = b.create<LLVM::BitcastOp>(
      getVoidPtrType(), b.create<LLVM::AddressOfOp>(global));
  auto toTy = LLVM::LLVMPointerType::get(LLVM::LLVMArrayType::get(i64Ty, 4));
  auto alloca = b.create<LLVM::BitcastOp>(
      getVoidPtrType(), b.create<LLVM::AllocaOp>(toTy, i32(1), 16));
  b.create<LLVM::MemcpyOp>(alloca, globalPtr, /*size=*/i64(8 * value.size()),
                           /*isVolatile=*/i1(false));

  auto nameVal = i32(attributeName);
  auto typeVal = i32(attributeType);
  auto valSize = i64(value.size());
  b.create<LLVM::CallOp>(
      ArrayRef<Type>{i32Ty}, getOrInsertSetAttributeFn(),
      ArrayRef<Value>{descriptor, nameVal, typeVal, valSize, alloca});
}

void BuildContext::setDescriptorAttribute(
    Value descriptor, cudnnBackendAttributeName_t attributeName,
    cudnnBackendAttributeType_t attributeType, Value lowered) {
  auto alloca = b.create<LLVM::BitcastOp>(getVoidPtrType(), lowered);

  auto nameVal = i32(attributeName);
  auto typeVal = i32(attributeType);
  auto valSize = i64(1);
  b.create<LLVM::CallOp>(
      ArrayRef<Type>{i32Ty}, getOrInsertSetAttributeFn(),
      ArrayRef<Value>{descriptor, nameVal, typeVal, valSize, alloca});
}

Value BuildContext::createDescriptor(int32_t kind) {
  auto fn = getOrInsertCreateDescriptorFn();
  Type i32Ty = b.getI32Type();
  auto ptrPtrTy = LLVM::LLVMPointerType::get(getVoidPtrType());
  auto alloca = b.create<LLVM::AllocaOp>(ptrPtrTy, i32(1), 8);
  b.create<LLVM::CallOp>(ArrayRef<Type>{i32Ty}, fn,
                         ArrayRef<Value>{i32(kind), alloca});
  // TODO: Check if should reload every time.
  return b.create<LLVM::LoadOp>(alloca);
}

void BuildContext::destroyDescriptor(Value descriptor) {
  Type i32Ty = b.getI32Type();
  auto fn = getOrInsertDestroyDescriptorFn();
  b.create<LLVM::CallOp>(ArrayRef<Type>{i32Ty}, fn,
                         ArrayRef<Value>{descriptor});
}

void BuildContext::finalizeDescriptor(Value descriptor) {
  Type i32Ty = b.getI32Type();
  auto fn = getOrInsertFinalizeDescriptorFn();
  b.create<LLVM::CallOp>(ArrayRef<Type>{i32Ty}, fn,
                         ArrayRef<Value>{descriptor});
}

int64_t elementTypeToDataType(Type t) {
  if (t.isF32())
    return CUDNN_DATA_FLOAT;
  if (t.isF64())
    return CUDNN_DATA_DOUBLE;
  if (t.isF16())
    return CUDNN_DATA_HALF;
  if (t.isSignlessInteger(8))
    return CUDNN_DATA_INT8;
  if (t.isSignlessInteger(32))
    return CUDNN_DATA_INT32;
  if (t.isUnsignedInteger(8))
    return CUDNN_DATA_UINT8;
  if (t.isBF16())
    return CUDNN_DATA_BFLOAT16;
  if (t.isSignlessInteger(64))
    return CUDNN_DATA_INT64;
  if (t.isSignedInteger(1))
    return CUDNN_DATA_BOOLEAN;
  // TODO: Error case.
  return -1;
}

Value BuildContext::createDescriptorFor(Value val) {
  if (tensorToDescriptor.count(val))
    return tensorToDescriptor[val];

  auto type = cast<cudnn::TensorDescType>(val.getType());
  auto desc = createDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
  setDescriptorAttribute(
      desc, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE,
      ArrayRef<int64_t>{elementTypeToDataType(type.getElementType())});
  setDescriptorAttribute(desc, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64,
                         type.getShape());
  setDescriptorAttribute(desc, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64,
                         type.getStride());
  setDescriptorAttribute(desc, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT,
                         CUDNN_TYPE_INT64,
                         ArrayRef<int64_t>(type.getAlignment()));
  // TODO: Improve UID.
  setDescriptorAttribute(desc, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64,
                         ArrayRef<int64_t>{++ctr});

  bool visible = llvm::any_of(val.getUsers(), [](Operation *user) {
    return isa<cudnn::BuildGraphOp>(user);
  });
  if (visible) {
    setDescriptorAttribute(desc, CUDNN_ATTR_TENSOR_IS_VIRTUAL,
                           CUDNN_TYPE_BOOLEAN, ArrayRef<int64_t>(visible));
  }

  // TODO: Missing by value, vector count/dimensions & reordering.

  finalizeDescriptor(desc);
  return tensorToDescriptor[val] = desc;
}

} // namespace

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
#define GEN_PASS_DEF_CONVERTCUDNNTOLLVMPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace {
struct ConvertCUDNNToLLVMPass
    : public impl::ConvertCUDNNToLLVMPassBase<ConvertCUDNNToLLVMPass> {
  using Base::Base;

  void runOnOperation() override;
};

LogicalResult lower(BuildContext &bctxt, cudnn::PointWiseReluOp pw) {
  // Create descriptor for operation:
  auto desc = bctxt.createDescriptor(CUDNN_BACKEND_POINTWISE_DESCRIPTOR);
  bctxt.setDescriptorAttribute(desc, CUDNN_ATTR_POINTWISE_MODE,
                               CUDNN_TYPE_POINTWISE_MODE,
                               ArrayRef<int64_t>(CUDNN_POINTWISE_RELU_FWD));
  bctxt.setDescriptorAttribute(
      desc, CUDNN_ATTR_POINTWISE_MATH_PREC, CUDNN_TYPE_DATA_TYPE,
      ArrayRef<int64_t>(elementTypeToDataType(pw.getComputeType())));
  // TODO: Should be op attribute.
  /*
  int64_t nan_progagation = CUDNN_NOT_PROPAGATE_NAN;
  bctxt.setDescriptorAttribute(desc, CUDNN_ATTR_POINTWISE_NAN_PROPAGATION,
                               CUDNN_TYPE_NAN_PROPOGATION, 1, &nan_propagation);
  */
  double lowerClip = pw.getLowerClip().convertToDouble();
  bctxt.setDescriptorAttribute(desc, CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP,
                               CUDNN_TYPE_DOUBLE, ArrayRef<double>(lowerClip));
  // TODO: Should be op attribute.
  double upper_clip = pw.getComputeType().isF32()
                          ? std::numeric_limits<float>::max()
                          : std::numeric_limits<double>::max();
  bctxt.setDescriptorAttribute(desc, CUDNN_ATTR_POINTWISE_RELU_UPPER_CLIP,
                               CUDNN_TYPE_DOUBLE, ArrayRef<double>(upper_clip));
  // TODO: Should be op attribute.
  double lower_clip_slope = 0.0;
  bctxt.setDescriptorAttribute(desc, CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP_SLOPE,
                               CUDNN_TYPE_DOUBLE,
                               ArrayRef<double>(lower_clip_slope));
  // Finalize descriptor.
  bctxt.finalizeDescriptor(desc);

  // Create node.
  auto nodeDesc =
      bctxt.createDescriptor(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR);
  auto xDesc = bctxt.createDescriptorFor(pw.getOperand());
  auto yDesc = bctxt.createDescriptorFor(pw.getResult());
  bctxt.setDescriptorAttribute(nodeDesc,
                               CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR,
                               CUDNN_TYPE_BACKEND_DESCRIPTOR, desc);
  bctxt.setDescriptorAttribute(nodeDesc, CUDNN_ATTR_OPERATION_POINTWISE_XDESC,
                               CUDNN_TYPE_BACKEND_DESCRIPTOR, xDesc);
  bctxt.setDescriptorAttribute(nodeDesc, CUDNN_ATTR_OPERATION_POINTWISE_YDESC,
                               CUDNN_TYPE_BACKEND_DESCRIPTOR, yDesc);
  // TODO: Should be op attribute.
  double alpha = 1.0;
  cudnnBackendAttributeType_t alphabetaType = CUDNN_TYPE_FLOAT;
  bctxt.setDescriptorAttribute(nodeDesc, CUDNN_ATTR_OPERATION_POINTWISE_ALPHA1,
                               alphabetaType, ArrayRef<double>(alpha));
  // TODO: Should be op attribute.
  double alpha2 = 1.0;
  bctxt.setDescriptorAttribute(nodeDesc, CUDNN_ATTR_OPERATION_POINTWISE_ALPHA2,
                               alphabetaType, ArrayRef<double>(alpha2));

  bctxt.finalizeDescriptor(nodeDesc);
  bctxt.tensorToDescriptor[pw.getResult()] = nodeDesc;

  return success();
}

LogicalResult lower(BuildContext &bctxt, cudnn::BuildGraphOp bgop) {
  bctxt.b.create<LLVM::ReturnOp>(bctxt.tensorToDescriptor[bgop.getOperand(0)]);
  return success();
}

// Converts a function consisting only of cudnn ops to a new builder function.
LogicalResult convertBuildAndExec(BuildContext &bctxt,
                                  cudnn::BuildAndExecGraphOp graph) {
  // Skip unless build graph terminator.
  if (!isa<cudnn::BuildGraphOp>(graph.getConstructor().front().getTerminator()))
    return success();

  // Create a build function.
  bctxt.b.setInsertionPointToStart(bctxt.module.getBody());
  auto llvmFnType = LLVM::LLVMFunctionType::get(bctxt.getVoidPtrType(),
                                                bctxt.getVoidPtrType(),
                                                /*isVarArg=*/false);
  bctxt.graphName = llvm::formatv("build_{0}", ++bctxt.ctr).str();
  auto fn = bctxt.b.create<LLVM::LLVMFuncOp>(bctxt.module.getLoc(),
                                             bctxt.graphName, llvmFnType);

  bctxt.b.setInsertionPointToStart(fn.addEntryBlock());
  for (auto &op : graph.getConstructor().front()) {
    // Tensor to descriptor mapping.
    bctxt.b.setLoc(op.getLoc());
    LogicalResult result =
        TypeSwitch<Operation *, LogicalResult>(&op)
            .Case<cudnn::PointWiseReluOp, cudnn::BuildGraphOp>(
                [&](auto pw) { return lower(bctxt, pw); })
            .Default([](Operation *op) { return success(); });
    if (failed(result))
      return failure();
  }

  // The memory lifetimes are a bit unclear with these, so just leak in v0.1.

  return success();
}

LogicalResult convertToLLVMCalls(ModuleOp module) {
  ImplicitLocOpBuilder b(module.getLoc(), module);
  BuildContext bctxt(module, b);

  auto res = module.walk([&](cudnn::BuildAndExecGraphOp g) {
    if (failed(convertBuildAndExec(bctxt, g)))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return failure(res.wasInterrupted());
}

} // namespace

void ConvertCUDNNToLLVMPass::runOnOperation() {
  if (failed(convertToLLVMCalls(getOperation())))
    return signalPassFailure();
}
