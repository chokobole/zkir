#ifndef ZKIR_DIALECT_FIELD_PIPELINES_PASSES_H_
#define ZKIR_DIALECT_FIELD_PIPELINES_PASSES_H_

#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Pass/PassOptions.h"

namespace mlir::zkir::field {

struct FieldToLLVMOptions : public PassPipelineOptions<FieldToLLVMOptions> {
  PassOptions::Option<bool> enableOpenMP{
      *this, "enable-openmp",
      llvm::cl::desc("Lowers parallel loops to OpenMP dialect"),
      llvm::cl::init(false)};

  PassOptions::Option<bool> bufferizeFunctionBoundaries{
      *this, "bufferize-function-boundaries",
      llvm::cl::desc("Bufferize function boundaries"), llvm::cl::init(false)};

  PassOptions::Option<bool> bufferResultsToOutParams{
      *this, "buffer-results-to-out-params",
      llvm::cl::desc("Buffer results to out params"), llvm::cl::init(true)};

  PassOptions::Option<bool> hoistStaticAllocs{
      *this, "hoist-static-allocs", llvm::cl::desc("Hoist static allocs"),
      llvm::cl::init(true)};

  // Projects out the options for `OneShotBufferizePass`.
  bufferization::OneShotBufferizePassOptions bufferizationOptions() const {
    bufferization::OneShotBufferizePassOptions opts{};
    opts.bufferizeFunctionBoundaries = bufferizeFunctionBoundaries;
    opts.functionBoundaryTypeConversion =
        bufferization::LayoutMapOption::IdentityLayoutMap;
    return opts;
  }

  // Projects out the options for `BufferResultsToOutParamsPass`.
  bufferization::BufferResultsToOutParamsPassOptions
  bufferResultsToOutParamsOptions() const {
    bufferization::BufferResultsToOutParamsPassOptions opts{};
    opts.hoistStaticAllocs = hoistStaticAllocs;
    return opts;
  }
};

// Adds the "field-to-llvm" pipeline to the `OpPassManager`.  This
// is the standard pipeline for taking field-based IR and lowering it
// to LLVM IR.
void buildFieldToLLVM(OpPassManager &pm, const FieldToLLVMOptions &options);

void registerFieldPipelines();

}  // namespace mlir::zkir::field

#endif  // ZKIR_DIALECT_FIELD_PIPELINES_PASSES_H_
