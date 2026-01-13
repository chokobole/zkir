/* Copyright 2026 The PrimeIR Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef PRIME_IR_UTILS_POWER_H_
#define PRIME_IR_UTILS_POWER_H_

#include "llvm/ADT/APInt.h" // IWYU pragma: keep
#include "mlir/Support/LLVM.h"
#include "zk_dtypes/include/bit_iterator.h"
#include "zk_dtypes/include/bit_traits_forward.h"

namespace zk_dtypes {

template <>
class BitTraits<mlir::APInt> {
public:
  static size_t GetNumBits(const mlir::APInt &value) {
    return value.getBitWidth();
  }

  static bool TestBit(const mlir::APInt &value, size_t index) {
    return value[index];
  }

  static void SetBit(mlir::APInt &value, size_t index, bool bitValue) {
    value.setBitVal(index, bitValue);
  }
};

} // namespace zk_dtypes

namespace mlir::prime_ir {

template <typename T>
T power(const T &value, const APInt &exponent) {
  auto ret = value.getOne();
  auto it = zk_dtypes::BitIteratorBE<APInt>::begin(&exponent, true);
  auto end = zk_dtypes::BitIteratorBE<APInt>::end(&exponent);
  while (it != end) {
    ret = ret.square();
    if (*it) {
      ret *= value;
    }
    ++it;
  }
  return ret;
}

} // namespace mlir::prime_ir

#endif // PRIME_IR_UTILS_POWER_H_
