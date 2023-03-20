add_mlir_conversion_library(MLIRCUDNNToLLVM
  CUDNNToLLVM.cpp

  DEPENDS
  MLIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRCUDNNDialect
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  )
