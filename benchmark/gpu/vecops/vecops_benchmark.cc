#include <climits>
#include <cstring>
#include <random>

#include "benchmark/BenchmarkUtils.h"
#include "benchmark/CudaUtils.h"
#include "benchmark/benchmark.h"
#include "cuda_runtime_api.h" // NOLINT(build/include_subdir)
#include "mlir/ExecutionEngine/MemRefUtils.h"
#include "mlir/Support/LLVM.h"

#define NUM_COEFFS (1 << 20)

namespace mlir::zkir::benchmark {
namespace {

using i64 = BigInt<1>;

extern "C" void _mlir_ciface_matvec_cpu(StridedMemRefType<i64, 2> *mat,
                                        StridedMemRefType<i64, 1> *vec,
                                        StridedMemRefType<i64, 1> *out);
extern "C" void _mlir_ciface_matvec_gpu(StridedMemRefType<i64, 2> *mat,
                                        StridedMemRefType<i64, 1> *vec,
                                        StridedMemRefType<i64, 1> *out);

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
void BM_matvec_benchmark(::benchmark::State &state) {
  OwningMemRef<i64, 2> hMat({NUM_COEFFS, 100}, {}, fillWithRandom);
  OwningMemRef<i64, 1> hVec({100}, {}, fillWithRandom);
  OwningMemRef<i64, 1> hOut({NUM_COEFFS}, {}, {});

  const size_t bytesMat = NUM_COEFFS * 100 * sizeof(i64);
  const size_t bytesVec = 100 * sizeof(i64);

  if constexpr (kIsGPU) {
    auto dMatBuf = makeCudaUnique<i64>(NUM_COEFFS * 100);
    auto dVecBuf = makeCudaUnique<i64>(100);
    auto dOutBuf = makeCudaUnique<i64>(NUM_COEFFS);

    CHECK_CUDA_ERROR(cudaMemcpy(dMatBuf.get(), hMat->data, bytesMat,
                                cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(dVecBuf.get(), hVec->data, bytesVec,
                                cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    StridedMemRefType<i64, 2> dMatRef{/*basePtr=*/dMatBuf.get(),
                                      /*data=*/dMatBuf.get(),
                                      /*offset=*/0,
                                      /*sizes=*/{NUM_COEFFS, 100},
                                      /*strides=*/{100, 1}};
    StridedMemRefType<i64, 1> dVecRef{/*basePtr=*/dVecBuf.get(),
                                      /*data=*/dVecBuf.get(),
                                      /*offset=*/0,
                                      /*sizes=*/{100},
                                      /*strides=*/{1}};
    StridedMemRefType<i64, 1> dOutRef{/*basePtr=*/dOutBuf.get(),
                                      /*data=*/dOutBuf.get(),
                                      /*offset=*/0,
                                      /*sizes=*/{NUM_COEFFS},
                                      /*strides=*/{1}};

    for (auto _ : state) {
      _mlir_ciface_matvec_gpu(&dMatRef, &dVecRef, &dOutRef);
    }
  } else {
    for (auto _ : state) {
      _mlir_ciface_matvec_cpu(&*hMat, &*hVec, &*hOut);
    }
  }
}

BENCHMARK_TEMPLATE(BM_matvec_benchmark, /*kIsGPU=*/false)
    ->Unit(::benchmark::kMillisecond)
    ->Name("matvec_cpu");
BENCHMARK_TEMPLATE(BM_matvec_benchmark, /*kIsGPU=*/true)
    ->Unit(::benchmark::kMillisecond)
    ->Name("matvec_gpu");

} // namespace
} // namespace mlir::zkir::benchmark

// clang-format off
// NOLINTBEGIN(whitespace/line_length)
//
// 2025-08-21T10:10:45+00:00
// Run on AMD Ryzen 9 9950X3D (32 X 624 MHz CPU s)
// CPU Caches:
//   L1 Data 48 KiB (x16)
//   L1 Instruction 32 KiB (x16)
//   L2 Unified 1024 KiB (x16)
//   L3 Unified 98304 KiB (x2)
// Load Average: 3.69, 3.33, 5.43
// -----------------------------------------------------
// Benchmark           Time             CPU   Iterations
// -----------------------------------------------------
// matvec_cpu       32.1 ms         29.6 ms           18
// matvec_gpu       8.14 ms         8.14 ms           87
// NOLINTEND()
