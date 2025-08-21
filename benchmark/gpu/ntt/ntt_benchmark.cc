#include <climits>
#include <cstring>
#include <random>

#include "benchmark/BenchmarkUtils.h"
#include "benchmark/CudaUtils.h"
#include "benchmark/benchmark.h"
#include "cuda_runtime_api.h" // NOLINT(build/include_subdir)
#include "gtest/gtest.h"
#include "mlir/ExecutionEngine/MemRefUtils.h"
#include "mlir/Support/LLVM.h"

#define NUM_COEFFS (1 << 20)

namespace mlir::zkir::benchmark {
namespace {

using i64 = BigInt<1>;

extern "C" void _mlir_ciface_ntt_cpu(StridedMemRefType<i64, 1> *input);
extern "C" void _mlir_ciface_intt_cpu(StridedMemRefType<i64, 1> *input);
extern "C" void _mlir_ciface_ntt_gpu(StridedMemRefType<i64, 1> *input);
extern "C" void _mlir_ciface_intt_gpu(StridedMemRefType<i64, 1> *input);

// `kPrime` = 9223372036836950017
const i64 kPrime = i64({9223372036836950017});

// Set up the random number generator.
std::mt19937_64 rng(std::random_device{}()); // NOLINT(whitespace/braces)
std::uniform_int_distribution<uint64_t> dist(0, UINT64_MAX);

// Set the element to random number in [0, kPrime).
void fillWithRandom(i64 &elem, ArrayRef<int64_t> coords) {
  elem = i64::randomLT(kPrime, rng, dist);
}

template <bool kIsGPU>
void BM_ntt_benchmark(::benchmark::State &state) {
  OwningMemRef<i64, 1> hInput(/*shape=*/{NUM_COEFFS}, /*shapeAlloc=*/{},
                              /*init=*/fillWithRandom);
  OwningMemRef<i64, 1> hTemp({NUM_COEFFS}, {});

  const size_t bytes = sizeof(i64) * NUM_COEFFS;

  if constexpr (kIsGPU) {
    auto dInputBuf = makeCudaUnique<i64>(NUM_COEFFS);
    auto dTmpBuf = makeCudaUnique<i64>(NUM_COEFFS);

    CHECK_CUDA_ERROR(cudaMemcpy(dInputBuf.get(), hInput->data, bytes,
                                cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    StridedMemRefType<i64, 1> dTmpRef{/*basePtr=*/dTmpBuf.get(),
                                      /*data=*/dTmpBuf.get(),
                                      /*offset=*/0,
                                      /*sizes=*/{NUM_COEFFS},
                                      /*strides=*/{1}};

    for (auto _ : state) {
      state.PauseTiming();
      CHECK_CUDA_ERROR(cudaMemcpy(dTmpBuf.get(), dInputBuf.get(), bytes,
                                  cudaMemcpyDeviceToDevice));
      CHECK_CUDA_ERROR(cudaDeviceSynchronize());
      state.ResumeTiming();

      _mlir_ciface_ntt_gpu(&dTmpRef);
    }

    _mlir_ciface_intt_gpu(&dTmpRef);

    // Copy back to host for a correctness check
    CHECK_CUDA_ERROR(
        cudaMemcpy(hTemp->data, dTmpBuf.get(), bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
  } else {
    for (auto _ : state) {
      state.PauseTiming();
      std::memcpy(hTemp->data, hInput->data, bytes);
      state.ResumeTiming();

      _mlir_ciface_ntt_cpu(&*hTemp);
    }

    _mlir_ciface_intt_cpu(&*hTemp);
  }

  // FIXME(batzor): The NTT benchmark is not working on GPU because the
  // `cuLaunchKernel` fails with `CUDA_ERROR_INVALID_VALUE`.
  if constexpr (!kIsGPU) {
    for (int i = 0; i < NUM_COEFFS; i++) {
      EXPECT_EQ((*hTemp)[i], (*hInput)[i]);
    }
  }
}

BENCHMARK_TEMPLATE(BM_ntt_benchmark, /*kIsGPU=*/false)
    ->Unit(::benchmark::kMillisecond)
    ->Name("ntt_cpu");
BENCHMARK_TEMPLATE(BM_ntt_benchmark, /*kIsGPU=*/true)
    ->Unit(::benchmark::kMillisecond)
    ->Name("ntt_gpu");

} // namespace
} // namespace mlir::zkir::benchmark

// clang-format off
// NOLINTBEGIN(whitespace/line_length)
//
// 2025-08-21T10:12:26+00:00
// Run on AMD Ryzen 9 9950X3D (32 X 624 MHz CPU s)
// CPU Caches:
//   L1 Data 48 KiB (x16)
//   L1 Instruction 32 KiB (x16)
//   L2 Unified 1024 KiB (x16)
//   L3 Unified 98304 KiB (x2)
// Load Average: 4.21, 3.76, 5.38
// -----------------------------------------------------
// Benchmark           Time             CPU   Iterations
// -----------------------------------------------------
// ntt_cpu          10.4 ms         9.70 ms           71
// ntt_gpu          ---- ms         ---- ms          ---
// NOLINTEND()
