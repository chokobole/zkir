#include <climits>
#include <random>

#include "benchmark/BenchmarkUtils.h"
#include "benchmark/benchmark.h"
#include "gtest/gtest.h"

#define NUM_SCALARMULS (1 << 20)

namespace mlir::zkir::benchmark {
namespace {

using i256 = BigInt<4>;

// `kPrime` =
// 21888242871839275222246405745257275088548364400416034343698204186575808495617
const i256 kPrimeBase = i256::fromHexString(
    "0x30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47");

const i256 kPrimeScalar = i256::fromHexString(
    "0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001");

// Fill the input with random numbers in [0, prime).
void fillWithRandom(Memref<i256> *input, const i256 &kPrime) {
  // Set up the random number generator.
  std::mt19937_64 rng(std::random_device{}());  // NOLINT(whitespace/braces)
  std::uniform_int_distribution<uint64_t> dist(0, UINT64_MAX);
  for (int i = 0; i < NUM_SCALARMULS; i++) {
    *input->pget(i, 0) = i256::randomLT(kPrime, rng, dist);
  }
}

// Fill the input with random numbers in [0, prime).
void fillWithRandomPoints(Memref<i256> *input, const i256 &kPrime) {
  // Set up the random number generator.
  std::mt19937_64 rng(std::random_device{}());  // NOLINT(whitespace/braces)
  std::uniform_int_distribution<uint64_t> dist(0, UINT64_MAX);
  for (int i = 0; i < NUM_SCALARMULS; i++) {
    *input->pget(i, 0) = i256::randomLT(kPrime, rng, dist);
    *input->pget(i, 1) = i256::randomLT(kPrime, rng, dist);
  }
}

extern "C" void _mlir_ciface_msm_serial(Memref<i256> *scalars,
                                        Memref<i256> *points);
extern "C" void _mlir_ciface_msm_parallel(Memref<i256> *scalars,
                                          Memref<i256> *points);

template <bool kIsParallel>
void BM_msm_benchmark(::benchmark::State &state) {
  Memref<i256> scalars(NUM_SCALARMULS, 1);
  fillWithRandom(&scalars, kPrimeScalar);
  Memref<i256> points(NUM_SCALARMULS, 2);
  fillWithRandomPoints(&points, kPrimeBase);

  for (auto _ : state) {
    if constexpr (kIsParallel) {
      _mlir_ciface_msm_parallel(&scalars, &points);
    } else {
      _mlir_ciface_msm_serial(&scalars, &points);
    }
  }
}

BENCHMARK_TEMPLATE(BM_msm_benchmark, /*kIsParallel=*/false)
    ->Iterations(20)
    ->Unit(::benchmark::kMillisecond)
    ->Name("msm_serial");

BENCHMARK_TEMPLATE(BM_msm_benchmark, /*kIsParallel=*/true)
    ->Iterations(20)
    ->Unit(::benchmark::kMillisecond)
    ->Name("msm_parallel");

}  // namespace
}  // namespace mlir::zkir::benchmark

// clang-format off
// NOLINTBEGIN(whitespace/line_length)

// 2025-08-07T01:40:36+00:00
// Run on AMD Ryzen 9 9950X3D (32 X 5501.43 MHz CPU s)
// CPU Caches:
//   L1 Data 48 KiB (x16)
//   L1 Instruction 32 KiB (x16)
//   L2 Unified 1024 KiB (x16)
//   L3 Unified 98304 KiB (x2)
// Load Average: 3.91, 4.78, 7.12
// ---------------------------------------------------------------------
// Benchmark                           Time             CPU   Iterations
// ---------------------------------------------------------------------
// msm_serial/iterations:20         2348 ms         2348 ms           20
// msm_parallel/iterations:20        276 ms          276 ms           20
// NOLINTEND()
