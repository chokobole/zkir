#include "zkir/Dialect/ArithExt/Conversions/SpecializeArithToAVX/SpecializeArithToAVX.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::zkir::arith_ext {

#define GEN_PASS_DEF_SPECIALIZEARITHTOAVX
#include "zkir/Dialect/ArithExt/Conversions/SpecializeArithToAVX/SpecializeArithToAVX.h.inc"

namespace {
inline bool isConstantSplat(Value value) {
  if (auto constantOp = value.getDefiningOp<arith::ConstantOp>()) {
    return isa<SplatElementsAttr>(constantOp.getValueAttr());
  }
  return false;
}

// Multiplies two vector<16xi32> operands using the vpmuludq instruction.
//
// vpmuludq performs extended multiplication on only the even lanes, producing
// vector<8xi64> results. This function bitcasts the results and returns a pair
// of vector<16xi32> values representing the extended products:
// - First: even lane extended products
// - Second: odd lane extended products
std::pair<Value, Value> mulExtendedByOddEven(ImplicitLocOpBuilder &b,
                                             Value lhsEven, Value lhsOdd,
                                             Value rhsEven, Value rhsOdd,
                                             bool toLowHi = false) {
  auto vecI32Type = VectorType::get(16, b.getI32Type());
  auto vecI64Type = VectorType::get(8, b.getI64Type());
  Value prodEven64 =
      b.create<LLVM::InlineAsmOp>(
           vecI64Type, ValueRange{lhsEven, rhsEven}, "vpmuludq $0, $1, $2",
           "=x,x,x", /*has_side_effects=*/false,
           /*is_align_stack=*/true, LLVM::TailCallKind::None,
           /*asm_dialect=*/
           LLVM::AsmDialectAttr::get(b.getContext(),
                                     LLVM::AsmDialect::AD_Intel),
           /*operand_attrs=*/ArrayAttr())
          .getResult(0);
  Value prodOdd64 =
      b.create<LLVM::InlineAsmOp>(
           vecI64Type, ValueRange{lhsOdd, rhsOdd}, "vpmuludq $0, $1, $2",
           "=x,x,x", /*has_side_effects=*/false,
           /*is_align_stack=*/true, LLVM::TailCallKind::None,
           /*asm_dialect=*/
           LLVM::AsmDialectAttr::get(b.getContext(),
                                     LLVM::AsmDialect::AD_Intel),
           /*operand_attrs=*/ArrayAttr())
          .getResult(0);

  // cast them to vector<16xi32> so even lanes are the low parts and odd
  // lanes are the high parts
  auto prodEven32 = b.create<vector::BitCastOp>(vecI32Type, prodEven64);
  auto prodOdd32 = b.create<vector::BitCastOp>(vecI32Type, prodOdd64);
  return {prodEven32, prodOdd32};
}

// Gathers the low parts of two vectors of 16 32-bit integers.
// [a₀, a₁, a₂, a₃, a₄, a₅, a₆, a₇, a₈, a₉, a₁₀, a₁₁, a₁₂, a₁₃, a₁₄, a₁₅]
// [b₀, b₁, b₂, b₃, b₄, b₅, b₆, b₇, b₈, b₉, b₁₀, b₁₁, b₁₂, b₁₃, b₁₄, b₁₅]
// => [a₀, b₀, a₂, b₂, a₄, b₄, a₆, b₆, a₈, b₈, a₁₀, b₁₀, a₁₂, b₁₂, a₁₄, b₁₄]
Value gatherLowsInterleaved(ImplicitLocOpBuilder &b, Value even, Value odd) {
  // 0b1010101010101010 = 0xAAAA
  Value constOddMask = b.create<LLVM::ConstantOp>(b.getI16Type(), 0xAAAA);
  auto vecI32Type = VectorType::get(16, b.getI32Type());

  // Construct vector<16xi32> with the low parts
  return b
      .create<LLVM::InlineAsmOp>(
          vecI32Type, ValueRange{even, constOddMask, odd},
          "vmovsldup $0 {$2}, $3", "=x,0,^Yk,x",
          /*has_side_effects=*/false,
          /*is_align_stack=*/true, LLVM::TailCallKind::None,
          /*asm_dialect=*/
          LLVM::AsmDialectAttr::get(b.getContext(), LLVM::AsmDialect::AD_Intel),
          /*operand_attrs=*/ArrayAttr())
      .getResult(0);
}

inline bool isGatherLowsResult(Value value) {
  if (auto inlineAsmOp = value.getDefiningOp<LLVM::InlineAsmOp>()) {
    return inlineAsmOp.getAsmString() == "vmovsldup $0 {$2}, $3";
  }
  return false;
}

// Gather the high parts of two vectors of 16 32-bit integers.
// [a₀, a₁, a₂, a₃, a₄, a₅, a₆, a₇, a₈, a₉, a₁₀, a₁₁, a₁₂, a₁₃, a₁₄, a₁₅]
// [b₀, b₁, b₂, b₃, b₄, b₅, b₆, b₇, b₈, b₉, b₁₀, b₁₁, b₁₂, b₁₃, b₁₄, b₁₅]
// => [a₁, b₁, a₃, b₃, a₅, b₅, a₇, b₇, a₉, b₉, a₁₁, b₁₁, a₁₃, b₁₃, a₁₅, b₁₅]
Value gatherHighsInterleaved(ImplicitLocOpBuilder &b, Value even, Value odd) {
  // 0b0101010101010101 = 0x5555
  Value constEvenMask = b.create<LLVM::ConstantOp>(b.getI16Type(), 0x5555);
  auto vecI32Type = VectorType::get(16, b.getI32Type());

  // Construct vector<16xi32> with the low parts
  return b
      .create<LLVM::InlineAsmOp>(
          vecI32Type, ValueRange{odd, constEvenMask, even},
          "vmovshdup $0 {$2}, $3", "=x,0,^Yk,x",
          /*has_side_effects=*/false,
          /*is_align_stack=*/true, LLVM::TailCallKind::None,
          /*asm_dialect=*/
          LLVM::AsmDialectAttr::get(b.getContext(), LLVM::AsmDialect::AD_Intel),
          /*operand_attrs=*/ArrayAttr())
      .getResult(0);
}

inline bool isGatherHighsResult(Value value) {
  if (auto inlineAsmOp = value.getDefiningOp<LLVM::InlineAsmOp>()) {
    return inlineAsmOp.getAsmString() == "vmovshdup $0 {$2}, $3";
  }
  return false;
}

// Duplicates the odd lanes of a vector<16xi32> to the even lanes.
// [a₀, a₁, a₂, a₃, a₄, a₅, a₆, a₇, a₈, a₉, a₁₀, a₁₁, a₁₂, a₁₃, a₁₄, a₁₅]
// => [a₁, a₁, a₃, a₃, a₅, a₅, a₇, a₇, a₉, a₉, a₁₁, a₁₁, a₁₃, a₁₃, a₁₅, a₁₅]
Value duplicateOddLanesToEven(ImplicitLocOpBuilder &b, Value vec) {
  auto vecI32Type = VectorType::get(16, b.getI32Type());
  return b.create<vector::ShuffleOp>(
      vecI32Type, vec, vec,
      b.getDenseI64ArrayAttr(
          {1, 1, 3, 3, 5, 5, 7, 7, 9, 9, 11, 11, 13, 13, 15, 15}));
}
} // namespace

struct SpecializeMulUIExtendedToAVX512
    : public OpConversionPattern<arith::MulUIExtendedOp> {
  explicit SpecializeMulUIExtendedToAVX512(MLIRContext *context)
      : OpConversionPattern<arith::MulUIExtendedOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::MulUIExtendedOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // if vector<16xi32> type, rewrite using vpmuludq, shuffle + vpmuludq
    if (auto vecType = dyn_cast<VectorType>(op.getLhs().getType())) {
      if (vecType.getElementType().isInteger(32) &&
          vecType.getNumElements() == 16) {
        ImplicitLocOpBuilder b(op.getLoc(), rewriter);

        Value lhsEven, lhsOdd;
        Value rhsEven, rhsOdd;
        if (isGatherLowsResult(adaptor.getLhs())) {
          lhsOdd = adaptor.getLhs()
                       .getDefiningOp<LLVM::InlineAsmOp>()
                       .getOperands()[2];
          lhsEven = adaptor.getLhs()
                        .getDefiningOp<LLVM::InlineAsmOp>()
                        .getOperands()[0];
        } else if (isGatherHighsResult(adaptor.getLhs())) {
          lhsOdd = adaptor.getLhs()
                       .getDefiningOp<LLVM::InlineAsmOp>()
                       .getOperands()[0];
          lhsEven = adaptor.getLhs()
                        .getDefiningOp<LLVM::InlineAsmOp>()
                        .getOperands()[2];
          lhsOdd = duplicateOddLanesToEven(b, lhsOdd);
          lhsEven = duplicateOddLanesToEven(b, lhsEven);
        } else {
          lhsOdd = duplicateOddLanesToEven(b, adaptor.getLhs());
          lhsEven = adaptor.getLhs();
        }

        if (isConstantSplat(adaptor.getRhs())) {
          rhsEven = adaptor.getRhs();
          rhsOdd = adaptor.getRhs();
        } else if (isGatherLowsResult(adaptor.getRhs())) {
          rhsOdd = adaptor.getRhs()
                       .getDefiningOp<LLVM::InlineAsmOp>()
                       .getOperands()[2];
          rhsEven = adaptor.getRhs()
                        .getDefiningOp<LLVM::InlineAsmOp>()
                        .getOperands()[0];
        } else if (isGatherHighsResult(adaptor.getRhs())) {
          rhsOdd = adaptor.getRhs()
                       .getDefiningOp<LLVM::InlineAsmOp>()
                       .getOperands()[0];
          rhsEven = adaptor.getRhs()
                        .getDefiningOp<LLVM::InlineAsmOp>()
                        .getOperands()[2];
        } else {
          rhsOdd = duplicateOddLanesToEven(b, adaptor.getRhs());
          rhsEven = adaptor.getRhs();
        }

        auto [prodEven32, prodOdd32] =
            mulExtendedByOddEven(b, lhsEven, lhsOdd, rhsEven, rhsOdd);

        Value prodLow = gatherLowsInterleaved(b, prodEven32, prodOdd32);
        Value prodHi = gatherHighsInterleaved(b, prodEven32, prodOdd32);

        rewriter.replaceOp(op, {prodLow, prodHi});
        return success();
      }
    }
    return failure();
  }
};

struct SpecializeMulIOpToAVX512 : public OpConversionPattern<arith::MulIOp> {
  explicit SpecializeMulIOpToAVX512(MLIRContext *context)
      : OpConversionPattern<arith::MulIOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::MulIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (auto vecType = dyn_cast<VectorType>(op.getLhs().getType())) {
      if (vecType.getElementType().isInteger(32) &&
          vecType.getNumElements() == 16) {
        ImplicitLocOpBuilder b(op.getLoc(), rewriter);
        Value lhsEven, lhsOdd;
        Value rhsEven, rhsOdd;
        if (isGatherLowsResult(adaptor.getLhs())) {
          lhsOdd = adaptor.getLhs()
                       .getDefiningOp<LLVM::InlineAsmOp>()
                       .getOperands()[2];
          lhsEven = adaptor.getLhs()
                        .getDefiningOp<LLVM::InlineAsmOp>()
                        .getOperands()[0];
        } else if (isGatherHighsResult(adaptor.getLhs())) {
          lhsOdd = adaptor.getLhs()
                       .getDefiningOp<LLVM::InlineAsmOp>()
                       .getOperands()[0];
          lhsEven = adaptor.getLhs()
                        .getDefiningOp<LLVM::InlineAsmOp>()
                        .getOperands()[2];
          lhsOdd = duplicateOddLanesToEven(b, lhsOdd);
          lhsEven = duplicateOddLanesToEven(b, lhsEven);
        } else {
          lhsOdd = duplicateOddLanesToEven(b, adaptor.getLhs());
          lhsEven = adaptor.getLhs();
        }

        if (isConstantSplat(adaptor.getRhs())) {
          rhsEven = adaptor.getRhs();
          rhsOdd = adaptor.getRhs();
        } else if (isGatherLowsResult(adaptor.getRhs())) {
          rhsOdd = adaptor.getRhs()
                       .getDefiningOp<LLVM::InlineAsmOp>()
                       .getOperands()[2];
          rhsEven = adaptor.getRhs()
                        .getDefiningOp<LLVM::InlineAsmOp>()
                        .getOperands()[0];
        } else if (isGatherHighsResult(adaptor.getRhs())) {
          rhsOdd = adaptor.getRhs()
                       .getDefiningOp<LLVM::InlineAsmOp>()
                       .getOperands()[0];
          rhsEven = adaptor.getRhs()
                        .getDefiningOp<LLVM::InlineAsmOp>()
                        .getOperands()[2];
          rhsOdd = duplicateOddLanesToEven(b, rhsOdd);
          rhsEven = duplicateOddLanesToEven(b, rhsEven);
        } else {
          rhsOdd = duplicateOddLanesToEven(b, adaptor.getRhs());
          rhsEven = adaptor.getRhs();
        }
        auto [prodEven32, prodOdd32] =
            mulExtendedByOddEven(b, lhsEven, lhsOdd, rhsEven, rhsOdd);
        Value prodLow = gatherLowsInterleaved(b, prodEven32, prodOdd32);
        rewriter.replaceOp(op, prodLow);
        return success();
      }
    }
    return failure();
  }
};

namespace {
#include "zkir/Dialect/ArithExt/Conversions/SpecializeArithToAVX/SpecializeArithToAVX.cpp.inc"
} // namespace

struct SpecializeArithToAVX
    : impl::SpecializeArithToAVXBase<SpecializeArithToAVX> {
  using SpecializeArithToAVXBase::SpecializeArithToAVXBase;

  void runOnOperation() override;
};

void SpecializeArithToAVX::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  ConversionTarget target(*context);
  target.addDynamicallyLegalOp<arith::MulUIExtendedOp>(
      [](arith::MulUIExtendedOp op) {
        // only specialize if the result is vector<16xi32>
        Type resultType = op.getResult(0).getType();
        if (auto vectorType = dyn_cast<VectorType>(resultType)) {
          return !(vectorType.getShape().size() == 1 &&
                   vectorType.getShape()[0] == 16 &&
                   vectorType.getElementType().isInteger(32));
        }
        return true;
      });
  target.addLegalDialect<LLVM::LLVMDialect>();
  target.addLegalDialect<vector::VectorDialect>();

  RewritePatternSet patterns(context);
  populateWithGenerated(patterns);
  patterns.add<SpecializeMulUIExtendedToAVX512>(context);
  patterns.add<SpecializeMulIOpToAVX512>(context);
  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

} // namespace mlir::zkir::arith_ext
