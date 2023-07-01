define dso_local noundef <16 x float> @foo(i1 noundef zeroext %scalarMask, i16 noundef zeroext %abMask, ptr nocapture noundef readonly %ptr, <16 x float> noundef %b, <16 x float> noundef %c) local_unnamed_addr {
entry:
  %extract.i = insertelement <4 x i1> <i1 poison, i1 false, i1 false, i1 false>, i1 %scalarMask, i64 0
  %0 = tail call <4 x float> @llvm.masked.load.v4f32.p0(ptr %ptr, i32 1, <4 x i1> %extract.i, <4 x float> <float 0.000000e+00, float poison, float poison, float poison>)
  %conv3 = sext i1 %scalarMask to i16
  %shuffle.i.i = shufflevector <4 x float> %0, <4 x float> poison, <16 x i32> zeroinitializer
  %1 = bitcast i16 %conv3 to <16 x i1>
  %2 = select <16 x i1> %1, <16 x float> %shuffle.i.i, <16 x float> zeroinitializer
  %3 = tail call <16 x float> @llvm.fma.v16f32(<16 x float> %2, <16 x float> %b, <16 x float> %c)
  %4 = select <16 x i1> %1, <16 x float> %3, <16 x float> zeroinitializer
   ret <16 x float> %4
  ;ret <16 x float> %shuffle.i.i
}

declare <4 x float> @llvm.masked.load.v4f32.p0(ptr nocapture, i32 immarg, <4 x i1>, <4 x float>) #1

declare <16 x float> @llvm.fma.v16f32(<16 x float>, <16 x float>, <16 x float>) #2

declare void @llvm.dbg.value(metadata, metadata, metadata) #3

