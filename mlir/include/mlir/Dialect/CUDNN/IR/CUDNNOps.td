//===- CUDNNOps.td - CUDNN dialect ops -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CUDNN_OPS
#define CUDNN_OPS

include "mlir/Dialect/CUDNN/IR/CUDNNDialect.td"
include "mlir/Dialect/CUDNN/IR/CUDNNTypes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Pointwise operations
// --------------------
def CUDNN_PointWiseAddOp : CUDNN_Op<"pointwise_add", [Pure]> {
    let summary = "Pointwise Add";

    let arguments = (ins
      CUDNN_TensorDescType:$input,
      TypeAttr:$element_type
    );
    let results = (outs CUDNN_TensorDescType:$res);

    let assemblyFormat = [{
        `(` $input `)` `type` `=` $element_type
	    attr-dict `:` type($input) `->` type($res)
    }];
}

def CUDNN_PointWiseReluOp : CUDNN_Op<"pointwise_relu", [Pure]> {
    let summary = "Pointwise Relu";

    let arguments = (ins
      CUDNN_TensorDescType:$input,
      TypeAttr:$element_type,
      F32Attr:$slope
    );
    let results = (outs CUDNN_TensorDescType:$res);

    let assemblyFormat = [{
        `(` $input `)` `type` `=` $element_type `slope` `=` $slope
	  attr-dict `:` type($input) `->` type($res)
    }];
}

// Convolutions
// ------------
def CUDNN_ConvolutionOp : CUDNN_Op<"convolution", [Pure]> {
    let summary = "Convolution";

    let arguments = (ins
      CUDNN_TensorDescType:$x,
      CUDNN_TensorDescType:$w,
      TypeAttr:$element_type,
      F32Attr:$alpha,
      F32Attr:$beta,
      I32Attr:$spatial_dim_count,
      DenseI64ArrayAttr:$spatial_stride,
      DenseI64ArrayAttr:$pre_padding,
      DenseI64ArrayAttr:$post_padding,
      DenseI64ArrayAttr:$dilation
    );
    let results = (outs CUDNN_TensorDescType:$y);

    let assemblyFormat = [{
      `(` $x `,` $w `)` `type` `=` $element_type
          `alpha` `=` $alpha
	  `beta` `=` $beta
	  `spatial_dim_count` `=` $spatial_dim_count
	  `spatial_stride` `=`  $spatial_stride
	  `pre_padding` `=` $pre_padding
	  `post_padding` `=` $post_padding
	  `dilation` `=` $dilation
	  attr-dict `:` type($x) `,` type($w) `->` type($y)
    }];
}

def CUDNN_CrossCorrelationOp : CUDNN_Op<"cross_correlation", [Pure]> {
    let summary = "Cross correlation";

    let arguments = (ins
      CUDNN_TensorDescType:$x,
      CUDNN_TensorDescType:$w,
      TypeAttr:$element_type,
      F32Attr:$alpha,
      F32Attr:$beta,
      I32Attr:$spatial_dim_count,
      DenseI64ArrayAttr:$spatial_stride,
      DenseI64ArrayAttr:$pre_padding,
      DenseI64ArrayAttr:$post_padding,
      DenseI64ArrayAttr:$dilation
    );
    let results = (outs CUDNN_TensorDescType:$y);

    let assemblyFormat = [{
      `(` $x `,` $w `)` `type` `=` $element_type
          `alpha` `=` $alpha
	  `beta` `=` $beta
	  `spatial_dim_count` `=` $spatial_dim_count
	  `spatial_stride` `=`  $spatial_stride
	  `pre_padding` `=` $pre_padding
	  `post_padding` `=` $post_padding
	  `dilation` `=` $dilation
	  attr-dict `:` type($x) `,` type($w) `->` type($y)
    }];
}

// MatMul
// ------
def CUDNN_MatMulOp : CUDNN_Op<"matmul", [Pure]> {
    let summary = "Matmul";

    let arguments = (ins
      CUDNN_TensorDescType:$a,
      CUDNN_TensorDescType:$b,
      TypeAttr:$element_type
    );
    let results = (outs CUDNN_TensorDescType:$c);

    let assemblyFormat = [{
        `(` $a `,` $b `)` `type` `=` $element_type
	  attr-dict `:` type($a) `,` type($b) `->` type($c)
    }];
}

// Reduction
// ---------
def CUDNN_ReductionOp : CUDNN_Op<"reduction", [Pure]> {
    let summary = "Reduction";

    let arguments = (ins
      CUDNN_TensorDescType:$x,
      // TODO: Shouldn't be enum.
      I32Attr:$reduction_op,
      TypeAttr:$element_type
    );
    let results = (outs CUDNN_TensorDescType:$y);

    let assemblyFormat = [{
        `(` $x `)` `type` `=` $element_type `reduction_op` `=` $reduction_op
	  attr-dict `:` type($x) `->` type($y)
    }];
}

// Utility functions.
// ------------------
def CUDNN_BuildGraphOp : CUDNN_Op<"build_graph", [Pure]> {
    let summary = "Matmul";

    let arguments = (ins
      CUDNN_HandleType:$handle,
      Variadic<CUDNN_TensorDescType>:$graph_results
    );
    let results = (outs CUDNN_OperationGraphType:$y);

    let assemblyFormat = [{
        $handle `(` $graph_results `)`
	  attr-dict `:` type($graph_results) `->` type($y)
    }];
}

#endif // CUDNN_OPS
