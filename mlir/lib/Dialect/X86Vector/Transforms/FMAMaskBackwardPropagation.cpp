//===- AVXTranspose.cpp - Lower Vector transpose to AVX -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements vector.transpose rewrites as AVX patterns for particular
// sizes of interest.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/Dialect/X86Vector/Transforms.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"

using namespace mlir;
using namespace mlir::vector;
using namespace mlir::x86vector;
using namespace mlir::x86vector::avx2;
using namespace mlir::x86vector::avx2::inline_asm;
using namespace mlir::x86vector::avx2::intrin;

static FailureOr<vector::BroadcastOp> getBroadcastOperand(vector::FMAOp fmaOp) {

  for (Value operand : fmaOp.getOperands()) {
    if (auto bcastOperand = operand.getDefiningOp<vector::BroadcastOp>())
      return bcastOperand;
  }

  return failure();
}

namespace {

class FMAMaskBackwardPropagation : public OpRewritePattern<vector::FMAOp> {
public:
  using OpRewritePattern<vector::FMAOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::FMAOp fmaOp,
                                PatternRewriter &rewriter) const override {

    PatternRewriter::InsertionGuard guard(rewriter);
    if (!fmaOp.getVectorType().getElementType().isInteger(32))
      failure();

    auto selectMask =
        dyn_cast<arith::SelectOp>(*fmaOp.getResult().getUsers().begin());
    if (!selectMask)
      return failure();

    auto maybeBcastOp = getBroadcastOperand(fmaOp);
    if (failed(maybeBcastOp))
      return failure();
    vector::BroadcastOp bcastOp = *maybeBcastOp;

    if (!bcastOp.getResult().hasOneUse())
      return failure();

    auto extractOp = bcastOp.getSource().getDefiningOp<vector::ExtractOp>();
    if (!extractOp)
      return failure();

    auto maskedLoad =
        extractOp.getVector().getDefiningOp<vector::MaskedLoadOp>();
    if (!maskedLoad)
      return failure();

    // if (!bcastOp.getResult().hasOneUse()) {
    //   TODO: We should or the mask of the multiple users.
    //
    //   // For broadcasts with multiple users, we "mask" the broadcast with a
    //   // select to make sure the unmasked scalar load is folded into the
    //   // broadcast in the backend. The broadcast then won't be folded into
    //   the
    //   // multiple users.
    //   rewriter.setInsertionPoint(fmaOp);
    //   auto bcastSelectMask = rewriter.create<arith::SelectOp>(
    //       selectMask.getLoc(), selectMask.getType(),
    //       selectMask.getCondition(), bcastOp.getResult(),
    //       selectMask.getFalseValue());
    //   rewriter.replaceUsesWithIf(
    //       bcastOp, bcastSelectMask, [&](OpOperand &operand) {
    //         return operand.getOwner() == fmaOp.getOperation();
    //       });

    //  //// Make sure the broadcasted operand is the second fma operand.
    //  // if (fmaOp.getOperand(0) == bcastSelectMask) {
    //  //   fmaOp.setOperand(0, fmaOp.getOperand(1));
    //  //   fmaOp.setOperand(1, bcastSelectMask);
    //  // }
    //}

    // We unmask the scalar load so that it's folded into the masked fma by the
    // backend.
    rewriter.setInsertionPointAfter(maskedLoad);
    rewriter.replaceOpWithNewOp<vector::LoadOp>(
        maskedLoad, maskedLoad.getVectorType(), maskedLoad.getBase(),
        maskedLoad.getIndices());

    return success();
  }
};

} // namespace

void mlir::x86vector::avx512::populateFMAMaskBackwardPropagationPatterns(
    RewritePatternSet &patterns) {
  patterns.add<FMAMaskBackwardPropagation>(patterns.getContext());
}
