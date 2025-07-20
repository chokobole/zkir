#include "zkir/Dialect/Field/Pipelines/Passes.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMPass.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/SCFToOpenMP/SCFToOpenMP.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "zkir/Dialect/Field/Conversions/FieldToModArith/FieldToModArith.h"
#include "zkir/Dialect/ModArith/Conversions/ModArithToArith/ModArithToArith.h"
#include "zkir/Dialect/TensorExt/Conversions/TensorExtToTensor/TensorExtToTensor.h"

//===----------------------------------------------------------------------===//
// Pipeline implementation.
//===----------------------------------------------------------------------===//

namespace mlir::zkir::field {

void buildFieldToLLVM(OpPassManager &pm, const FieldToLLVMOptions &options) {
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::createConvertElementwiseToLinalgPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::createLinalgElementwiseOpFusionPass());
  pm.addPass(createFieldToModArith());
  pm.addPass(createCanonicalizerPass());

  pm.addPass(mod_arith::createModArithToArith());
  pm.addPass(createCanonicalizerPass());

  pm.addPass(createLowerAffinePass());

  pm.addPass(tensor_ext::createTensorExtToTensor());

  pm.addPass(bufferization::createOneShotBufferizePass(
      options.bufferizationOptions()));

  if (options.bufferResultsToOutParams) {
    pm.addPass(bufferization::createBufferResultsToOutParamsPass(
        options.bufferResultsToOutParamsOptions()));
  }

  pm.addNestedPass<func::FuncOp>(createConvertLinalgToParallelLoopsPass());

  if (options.enableOpenMP) {
    pm.addPass(createConvertSCFToOpenMPPass());
  }

  pm.addNestedPass<func::FuncOp>(memref::createExpandStridedMetadataPass());
  pm.addPass(createLowerAffinePass());
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(createSCFToControlFlowPass());
  pm.addPass(createConvertToLLVMPass());
  pm.addPass(createCanonicalizerPass());
}

//===----------------------------------------------------------------------===//
// Pipeline registration.
//===----------------------------------------------------------------------===//

void registerFieldPipelines() {
  PassPipelineRegistration<FieldToLLVMOptions>(
      "field-to-llvm",
      "The standard pipeline for taking field-agnostic IR using the"
      " field type, and lowering it to LLVM IR with concrete"
      " representations and algorithms for fields.",
      buildFieldToLLVM);
}

}  // namespace mlir::zkir::field
