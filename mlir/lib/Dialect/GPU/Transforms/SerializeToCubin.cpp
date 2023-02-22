//===- LowerGPUToCUBIN.cpp - Convert GPU kernel to CUBIN blob -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass that serializes a gpu module into CUBIN blob and
// adds that blob as a string attribute of the module.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/GPU/Transforms/Passes.h"

#if MLIR_GPU_TO_CUBIN_PASS_ENABLE
#include "mlir/Pass/Pass.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Linker/Linker.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/Internalize.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

#include <cuda.h>

static llvm::cl::opt<std::string> libDevicePath(
    "cuda-libdevice-path", llvm::cl::desc("path to libdevice.bc"),
    llvm::cl::init(""));

namespace llvm {
class FunctionPass;
FunctionPass *createNVVMIntrRangePass(unsigned int SmVersion);
FunctionPass *createNVVMReflectPass(unsigned int SmVersion);
}  // namespace llvm
using namespace mlir;

static void emitCudaError(const llvm::Twine &expr, const char *buffer,
                          CUresult result, Location loc) {
  const char *error;
  cuGetErrorString(result, &error);
  emitError(loc, expr.concat(" failed with error code ")
                     .concat(llvm::Twine{error})
                     .concat("[")
                     .concat(buffer)
                     .concat("]"));
}

#define RETURN_ON_CUDA_ERROR(expr)                                             \
  do {                                                                         \
    if (auto status = (expr)) {                                                \
      emitCudaError(#expr, jitErrorBuffer, status, loc);                       \
      return {};                                                               \
    }                                                                          \
  } while (false)

namespace {
class SerializeToCubinPass
    : public PassWrapper<SerializeToCubinPass, gpu::SerializeToBlobPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SerializeToCubinPass)

  SerializeToCubinPass(StringRef triple = "nvptx64-nvidia-cuda",
                       StringRef chip = "sm_35", StringRef features = "+ptx60");

  StringRef getArgument() const override { return "gpu-to-cubin"; }
  StringRef getDescription() const override {
    return "Lower GPU kernel function to CUBIN binary annotations";
  }

protected:
  LogicalResult optimizeLlvm(llvm::Module &llvmModule,
                                     llvm::TargetMachine &targetMachine) override;

private:
  void getDependentDialects(DialectRegistry &registry) const override;

  // Serializes PTX to CUBIN.
  std::unique_ptr<std::vector<char>>
  serializeISA(const std::string &isa) override;
};
} // namespace

// Sets the 'option' to 'value' unless it already has a value.
static void maybeSetOption(Pass::Option<std::string> &option, StringRef value) {
  if (!option.hasValue())
    option = value.str();
}

SerializeToCubinPass::SerializeToCubinPass(StringRef triple, StringRef chip,
                                           StringRef features) {
  maybeSetOption(this->triple, triple);
  maybeSetOption(this->chip, chip);
  maybeSetOption(this->features, features);
}

void SerializeToCubinPass::getDependentDialects(
    DialectRegistry &registry) const {
  registerNVVMDialectTranslation(registry);
  gpu::SerializeToBlobPass::getDependentDialects(registry);
}

LogicalResult SerializeToCubinPass::optimizeLlvm(llvm::Module &module,
                                     llvm::TargetMachine &targetMachine) {
 llvm::Linker linker(module);

 std::string path(libDevicePath);
 if(path.empty())
  return success();
 llvm::SMDiagnostic Err;
 std::unique_ptr<llvm::Module> bitcodeModule =
     llvm::parseIRFile(path, Err, module.getContext());

 if (!bitcodeModule) {
    llvm::errs() << "failed to parse CUDA libdevice bitcode";
    return failure();
  }
  // Ignore the data layout of the module we're importing. This avoids a
  // warning from the linker.
  bitcodeModule->setDataLayout(module.getDataLayout());
  linker.linkInModule(
      std::move(bitcodeModule), llvm::Linker::Flags::LinkOnlyNeeded,
      [](llvm::Module &M, const llvm::StringSet<> &GVS) {
        llvm::internalizeModule(M, [&GVS](const llvm::GlobalValue &GV) {
          return !GV.hasName() || (GVS.count(GV.getName()) == 0);
        });
      });

   // run optimizer. Code from IREE CUDA target.

   
  // Workaround run those passed ahead as they are temporarily disabled in NVPTX
  // target.
  llvm::legacy::PassManager legacyPM;
  legacyPM.add(llvm::createNVVMIntrRangePass(35));
  legacyPM.add(llvm::createNVVMReflectPass(35));
  legacyPM.run(module);

  llvm::LoopAnalysisManager lam;
  llvm::FunctionAnalysisManager fam;
  llvm::CGSCCAnalysisManager cgam;
  llvm::ModuleAnalysisManager mam;

  fam.registerPass([&] { return targetMachine.getTargetIRAnalysis(); });

  llvm::PipelineTuningOptions pto;
  pto.SLPVectorization = false;

  llvm::PassInstrumentationCallbacks pic;

  llvm::StandardInstrumentations si(module.getContext(), false);
  si.registerCallbacks(pic, &fam);

  llvm::PassBuilder pb(&targetMachine, pto, std::nullopt, &pic);
  pb.registerModuleAnalyses(mam);
  pb.registerCGSCCAnalyses(cgam);
  pb.registerFunctionAnalyses(fam);
  pb.registerLoopAnalyses(lam);
  pb.crossRegisterProxies(lam, fam, cgam, mam);

  llvm::OptimizationLevel ol = llvm::OptimizationLevel::O2;

  llvm::ModulePassManager mpm;
  mpm.addPass(llvm::VerifierPass());
  mpm.addPass(pb.buildPerModuleDefaultPipeline(ol));
  mpm.addPass(llvm::VerifierPass());

  mpm.run(module, mam);   

 return success();
}

std::unique_ptr<std::vector<char>>
SerializeToCubinPass::serializeISA(const std::string &isa) {
  Location loc = getOperation().getLoc();
  char jitErrorBuffer[4096] = {0};

  RETURN_ON_CUDA_ERROR(cuInit(0));

  // Linking requires a device context.
  CUdevice device;
  RETURN_ON_CUDA_ERROR(cuDeviceGet(&device, 0));
  CUcontext context;
  RETURN_ON_CUDA_ERROR(cuCtxCreate(&context, 0, device));
  CUlinkState linkState;

  CUjit_option jitOptions[] = {CU_JIT_ERROR_LOG_BUFFER,
                               CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES};
  void *jitOptionsVals[] = {jitErrorBuffer,
                            reinterpret_cast<void *>(sizeof(jitErrorBuffer))};

  RETURN_ON_CUDA_ERROR(cuLinkCreate(2,              /* number of jit options */
                                    jitOptions,     /* jit options */
                                    jitOptionsVals, /* jit option values */
                                    &linkState));

  auto kernelName = getOperation().getName().str();
  RETURN_ON_CUDA_ERROR(cuLinkAddData(
      linkState, CUjitInputType::CU_JIT_INPUT_PTX,
      const_cast<void *>(static_cast<const void *>(isa.c_str())), isa.length(),
      kernelName.c_str(), 0, /* number of jit options */
      nullptr,               /* jit options */
      nullptr                /* jit option values */
      ));

  void *cubinData;
  size_t cubinSize;
  RETURN_ON_CUDA_ERROR(cuLinkComplete(linkState, &cubinData, &cubinSize));

  char *cubinAsChar = static_cast<char *>(cubinData);
  auto result =
      std::make_unique<std::vector<char>>(cubinAsChar, cubinAsChar + cubinSize);

  // This will also destroy the cubin data.
  RETURN_ON_CUDA_ERROR(cuLinkDestroy(linkState));
  RETURN_ON_CUDA_ERROR(cuCtxDestroy(context));

  return result;
}

// Register pass to serialize GPU kernel functions to a CUBIN binary annotation.
void mlir::registerGpuSerializeToCubinPass() {
  PassRegistration<SerializeToCubinPass> registerSerializeToCubin([] {
    // Initialize LLVM NVPTX backend.
    LLVMInitializeNVPTXTarget();
    LLVMInitializeNVPTXTargetInfo();
    LLVMInitializeNVPTXTargetMC();
    LLVMInitializeNVPTXAsmPrinter();

    return std::make_unique<SerializeToCubinPass>();
  });
}

std::unique_ptr<Pass> mlir::createGpuSerializeToCubinPass(StringRef triple,
                                                          StringRef arch,
                                                          StringRef features) {
  return std::make_unique<SerializeToCubinPass>(triple, arch, features);
}

#else  // MLIR_GPU_TO_CUBIN_PASS_ENABLE
void mlir::registerGpuSerializeToCubinPass() {}
#endif // MLIR_GPU_TO_CUBIN_PASS_ENABLE
