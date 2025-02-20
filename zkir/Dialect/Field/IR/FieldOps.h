#ifndef ZKIR_DIALECT_FIELD_IR_FIELDOPS_H_
#define ZKIR_DIALECT_FIELD_IR_FIELDOPS_H_

// NOLINTBEGIN(misc-include-cleaner): Required to define FieldOps
#include "mlir/include/mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/include/mlir/Interfaces/InferTypeOpInterface.h"  // from @llvm-project
#include "zkir/Dialect/Field/IR/FieldAttributes.h"
#include "zkir/Dialect/Field/IR/FieldDialect.h"
#include "zkir/Dialect/Field/IR/FieldTypes.h"
// NOLINTEND(misc-include-cleaner)

#define GET_OP_CLASSES
#include "zkir/Dialect/Field/IR/FieldOps.h.inc"

#endif  // ZKIR_DIALECT_FIELD_IR_FIELDOPS_H_
