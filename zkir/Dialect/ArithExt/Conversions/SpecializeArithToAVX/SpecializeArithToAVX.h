#ifndef ZKIR_DIALECT_ARITHEXT_CONVERSIONS_SPECIALIZEARITHTOAVX_SPECIALIZEARITHTOAVX_H_
#define ZKIR_DIALECT_ARITHEXT_CONVERSIONS_SPECIALIZEARITHTOAVX_SPECIALIZEARITHTOAVX_H_

// IWYU pragma: begin_keep
// Headers needed for SpecializeArithToAVX.h.inc
#include "mlir/Pass/Pass.h"
// IWYU pragma: end_keep

namespace mlir::zkir::arith_ext {

#define GEN_PASS_DECL
#include "zkir/Dialect/ArithExt/Conversions/SpecializeArithToAVX/SpecializeArithToAVX.h.inc"

#define GEN_PASS_REGISTRATION
#include "zkir/Dialect/ArithExt/Conversions/SpecializeArithToAVX/SpecializeArithToAVX.h.inc" // NOLINT(build/include)

} // namespace mlir::zkir::arith_ext

// NOLINTNEXTLINE(whitespace/line_length)
#endif // ZKIR_DIALECT_ARITHEXT_CONVERSIONS_SPECIALIZEARITHTOAVX_SPECIALIZEARITHTOAVX_H_
