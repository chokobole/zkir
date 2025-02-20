#ifndef ZKIR_DIALECT_FIELD_IR_FIELDOPS_TD_
#define ZKIR_DIALECT_FIELD_IR_FIELDOPS_TD_

include "zkir/Dialect/Field/IR/FieldAttributes.td"
include "zkir/Dialect/Field/IR/FieldDialect.td"
include "zkir/Dialect/Field/IR/FieldTypes.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def PrimeField_ConstantOp : Op<Field_Dialect, "pf.constant", [Pure, InferTypeOpAdaptor]> {
  let summary = "Define a constant prime field element via an attribute";
  let description = [{
    Example:

    ```mlir
    %0 = field.pf.constant 123 : !field.pf<7>
    ```
  }];

  let arguments = (ins Field_PrimeFieldAttr:$value);
  let results = (outs Field_PrimeFieldType:$output);
  let hasCustomAssemblyFormat = 1;

  let builders = [
    OpBuilder<(ins "PrimeFieldType":$ty, "const llvm::APInt":$value), [{
      return build($_builder, $_state, ty, PrimeFieldAttr::get(ty, value));
    }]>
  ];
}

#endif  // ZKIR_DIALECT_FIELD_IR_FIELDOPS_TD_
