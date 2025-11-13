#ifndef ZKIR_DIALECT_ARITHEXT_CONVERSIONS_SPECIALIZEARITHTOAVX_SPECIALIZEARITHTOAVX_TD_
#define ZKIR_DIALECT_ARITHEXT_CONVERSIONS_SPECIALIZEARITHTOAVX_SPECIALIZEARITHTOAVX_TD_

include "mlir/Dialect/LLVMIR/LLVMOps.td"
include "mlir/IR/PatternBase.td"
include "mlir/Pass/PassBase.td"

def SpecializeArithToAVX : Pass<"specialize-arith-to-avx", "ModuleOp"> {
  let summary = "Specialize some `arith` ops to assembly";
  let description = [{
    This pass specializes some `arith` ops to assembly.
  }];
  let dependentDialects = [
    "arith::ArithDialect",
    "LLVM::LLVMDialect",
    "vector::VectorDialect",
  ];
}

#endif // ZKIR_DIALECT_ARITHEXT_CONVERSIONS_SPECIALIZEARITHTOAVX_SPECIALIZEARITHTOAVX_TD_
