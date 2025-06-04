#ifndef ZKIR_DIALECT_ELLIPTICCURVE_CONVERSIONS_ELLIPTICCURVETOFIELD_MSM_PIPPENGERS_PIPPENGERS_H_
#define ZKIR_DIALECT_ELLIPTICCURVE_CONVERSIONS_ELLIPTICCURVETOFIELD_MSM_PIPPENGERS_PIPPENGERS_H_

#include <cmath>

namespace mlir::zkir::elliptic_curve {

// The result of this function is only approximately `ln(a)`.
// See https://github.com/scipr-lab/zexe/issues/79#issue-556220473
constexpr static size_t lnWithoutFloats(size_t a) {
  // log2(a) * ln(2)
  return std::log2(a) * 69 / 100;
}

constexpr size_t computeWindowsBits(size_t size) {
  if (size < 32) {
    return 3;
  } else {
    return lnWithoutFloats(size) + 2;
  }
}

constexpr size_t computeWindowsCount(size_t scalarBitWidth,
                                     size_t bitsPerWindow) {
  return (scalarBitWidth + bitsPerWindow - 1) / bitsPerWindow;
}

}  // namespace mlir::zkir::elliptic_curve

#endif  // ZKIR_DIALECT_ELLIPTICCURVE_CONVERSIONS_ELLIPTICCURVETOFIELD_MSM_PIPPENGERS_PIPPENGERS_H_
