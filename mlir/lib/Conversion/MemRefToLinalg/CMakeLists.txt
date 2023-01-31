add_mlir_conversion_library(MLIRMemRefToLinalg
  MemRefToLinalg.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Linalg
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/IR

  DEPENDS
  MLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRSupport
  MLIRTransformUtils
  MLIRMemRefDialect
  )
