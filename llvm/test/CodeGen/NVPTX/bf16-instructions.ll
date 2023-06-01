; RUN: llc < %s -march=nvptx64 -mcpu=sm_80 -mattr=+ptx70 | FileCheck %s
; RUN: %if ptxas-11.0 %{ llc < %s -march=nvptx64 -mcpu=sm_80 -mattr=+ptx70 | %ptxas-verify -arch=sm_80 %}


; CHECK-LABEL: test_fadd(
; CHECK-DAG:  ld.param.b16    [[A:%h[0-9]+]], [test_fadd_param_0];
; CHECK-DAG:  ld.param.b16    [[B:%h[0-9]+]], [test_fadd_param_1];
; CHECK-NEXT: add.rn.bf16     [[R:%f[0-9]+]], [[A]], [[B]];
; CHECK-NEXT: st.param.b16    [func_retval0+0], [[R]];
; CHECK-NEXT: ret;

define bfloat @test_fadd(bfloat %0, bfloat %1) {
  %3 = fadd bfloat %0, %1                                         
  ret bfloat %3
}

; CHECK-LABEL: test_fsub(
; CHECK-DAG:  ld.param.b16    [[A:%h[0-9]+]], [test_fsub_param_0];
; CHECK-DAG:  ld.param.b16    [[B:%h[0-9]+]], [test_fsub_param_1];
; CHECK-NEXT: sub.rn.bf16     [[R:%f[0-9]+]], [[A]], [[B]];
; CHECK-NEXT: st.param.b16    [func_retval0+0], [[R]];
; CHECK-NEXT: ret;

define bfloat @test_fsub(bfloat %0, bfloat %1) {
  %3 = fsub bfloat %0, %1                                         
  ret bfloat %3
}

; CHECK-LABEL: test_faddx2(
; CHECK-DAG:  ld.param.b32    [[A:%hh[0-9]+]], [test_faddx2_param_0];
; CHECK-DAG:  ld.param.b32    [[B:%hh[0-9]+]], [test_faddx2_param_1];
; CHECK-NEXT: add.rn.bf16x2   [[R:%f[0-9]+]], [[A]], [[B]];

; CHECK:      st.param.b32    [func_retval0+0], [[R]];
; CHECK:      ret;

define <2 x bfloat> @test_faddx2(<2 x bfloat> %a, <2 x bfloat> %b) #0 {
  %r = fadd <2 x bfloat> %a, %b
  ret <2 x bfloat> %r
}

; CHECK-LABEL: test_fsubx2(
; CHECK-DAG:  ld.param.b32    [[A:%hh[0-9]+]], [test_fsubx2_param_0];
; CHECK-DAG:  ld.param.b32    [[B:%hh[0-9]+]], [test_fsubx2_param_1];
; CHECK-NEXT: sub.rn.bf16x2   [[R:%f[0-9]+]], [[A]], [[B]];

; CHECK:      st.param.b32    [func_retval0+0], [[R]];
; CHECK:      ret;

define <2 x bfloat> @test_fsubx2(<2 x bfloat> %a, <2 x bfloat> %b) #0 {
  %r = fsub <2 x bfloat> %a, %b
  ret <2 x bfloat> %r
}

; CHECK-LABEL: test_fmulx2(
; CHECK-DAG:  ld.param.b32    [[A:%hh[0-9]+]], [test_fmulx2_param_0];
; CHECK-DAG:  ld.param.b32    [[B:%hh[0-9]+]], [test_fmulx2_param_1];
; CHECK-NEXT: mul.rn.bf16x2   [[R:%f[0-9]+]], [[A]], [[B]];

; CHECK:      st.param.b32    [func_retval0+0], [[R]];
; CHECK:      ret;

define <2 x bfloat> @test_fmul(<2 x bfloat> %a, <2 x bfloat> %b) #0 {
  %r = fmul <2 x bfloat> %a, %b
  ret <2 x bfloat> %r
}

; CHECK-LABEL: test_fdiv(
; CHECK-DAG:  ld.param.b32    [[A:%hh[0-9]+]], [test_fdiv_param_0];
; CHECK-DAG:  ld.param.b32    [[B:%hh[0-9]+]], [test_fdiv_param_1];
; CHECK-DAG:  mov.b32         {[[A0:%h[0-9]+]], [[A1:%h[0-9]+]]}, [[A]]
; CHECK-DAG:  mov.b32         {[[B0:%h[0-9]+]], [[B1:%h[0-9]+]]}, [[B]]
; CHECK-DAG:  cvt.f32.bf16     [[FA0:%f[0-9]+]], [[A0]];
; CHECK-DAG:  cvt.f32.bf16     [[FA1:%f[0-9]+]], [[A1]];
; CHECK-DAG:  cvt.f32.bf16     [[FB0:%f[0-9]+]], [[B0]];
; CHECK-DAG:  cvt.f32.bf16     [[FB1:%f[0-9]+]], [[B1]];
; CHECK-DAG:  div.rn.f32      [[FR0:%f[0-9]+]], [[FA0]], [[FB0]];
; CHECK-DAG:  div.rn.f32      [[FR1:%f[0-9]+]], [[FA1]], [[FB1]];
; CHECK-DAG:  cvt.rn.bf16.f32  [[R0:%h[0-9]+]], [[FR0]];
; CHECK-DAG:  cvt.rn.bf16.f32  [[R1:%h[0-9]+]], [[FR1]];
; CHECK-NEXT: mov.b32         [[R:%hh[0-9]+]], {[[R0]], [[R1]]}
; CHECK-NEXT: st.param.b32    [func_retval0+0], [[R]];
; CHECK-NEXT: ret;

define <2 x bfloat> @test_fdiv(<2 x bfloat> %a, <2 x bfloat> %b) #0 {
  %r = fdiv <2 x bfloat> %a, %b
  ret <2 x bfloat> %r
}
