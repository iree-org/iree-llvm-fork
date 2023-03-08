func.func @foo(%arg0 : !cudnn.tensor_desc<4x24x31x31xbf16, alignment=16,
                            stride=[23064, 1, 744, 24]> loc("x"),
			%graph: !cudnn.operation_graph, %sta : !cudnn.status) -> !cudnn.status {
  %10 = cudnn.pointwise_relu(%arg0) type=f32 slope = 0.01 :
    !cudnn.tensor_desc<4x24x31x31xbf16, alignment=16,stride=[23064, 1, 744, 24]> ->
    !cudnn.tensor_desc<4x24x31x31xbf16, alignment=16, stride=[23064, 1, 744, 24]>
  %y = cudnn.convolution(%10, %arg0) type=f32 alpha=1.0 beta=0.0 spatial_dim_count=4
    spatial_stride=[10,20] pre_padding=[3,2] post_padding=[4,4] dilation=[2] :
      !cudnn.tensor_desc<4x24x31x31xbf16, alignment=16,stride=[23064, 1, 744, 24]>,
       !cudnn.tensor_desc<4x24x31x31xbf16, alignment=16, stride=[23064, 1, 744, 24]> ->
         !cudnn.tensor_desc<4x24x31x31xbf16, alignment=16, stride=[23064, 1, 744, 24]>
  return %sta : !cudnn.status
}

func.func @execute(%graph: !cudnn.operation_graph, %plan: !cudnn.execution_plan, %workspace: memref<?xi8>,
     %x: memref<4x31x31x24xbf16>, %y: memref<32x3x3x24xbf16>) -> !cudnn.status {
  %0 = cudnn.execute %graph with %plan[%workspace] (%x, %y) : 
     (memref<4x31x31x24xbf16>, memref<32x3x3x24xbf16>) [memref<?xi8>] -> !cudnn.status
  return %0 : !cudnn.status
}


