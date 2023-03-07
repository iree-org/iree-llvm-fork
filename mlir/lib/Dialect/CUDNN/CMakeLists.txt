add_mlir_dialect_library(MLIRCUDNN
        CUDNNDialect.cpp
        CUDNNOps.cpp
        CUDNNTypes.cpp

        DEPENDS
        MLIRCUDNNOpsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
        MLIRInferTypeOpInterface
	)
