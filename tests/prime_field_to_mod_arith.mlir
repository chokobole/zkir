// RUN: tools/zkir-opt -prime-field-to-mod-arith --split-input-file %s | FileCheck %s --enable-var-scope
!PF1 = !field.pf<3:i32>
!PFv = tensor<4x!PF1>

// CHECK-LABEL: @test_lower_constant
func.func @test_lower_constant() {
  // CHECK: %[[RES:.*]] = mod_arith.constant 4 : !Z3_i32_
  %res = field.pf.constant 4 : !PF1
  return
}

// CHECK-LABEL: @test_lower_add
func.func @test_lower_add() -> !PF1 {
  // CHECK: %[[C0:.*]] = mod_arith.constant 4 : !Z3_i32_
  %c0 = field.pf.constant 4 : !PF1
  // CHECK: %[[RES:.*]] = mod_arith.add %[[C0]], %[[C0]] : !Z3_i32_
  %res = field.pf.add %c0, %c0 : !PF1
  return %res : !PF1
}

// CHECK-LABEL: @test_lower_add_vec
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]], %[[RHS:.*]]: [[T]]) -> [[T]] {
func.func @test_lower_add_vec(%lhs : !PFv, %rhs : !PFv) -> !PFv {
  // CHECK-NOT: field.pf.add
  // CHECK: %[[RES:.*]] = mod_arith.add %[[LHS]], %[[RHS]] : tensor<4x!Z3_i32_>
  %res = field.pf.add %lhs, %rhs : !PFv
  // CHECK: return %[[RES]] : [[T]]
  return %res : !PFv
}

// CHECK-LABEL: @test_lower_sub
func.func @test_lower_sub() -> !PF1 {
  // CHECK: %[[C0:.*]] = mod_arith.constant 4 : !Z3_i32_
  %c0 = field.pf.constant 4 : !PF1
  // CHECK: %[[C1:.*]] = mod_arith.constant 5 : !Z3_i32_
  %c1 = field.pf.constant 5 : !PF1
  // CHECK: %[[RES:.*]] = mod_arith.sub %[[C0]], %[[C1]] : !Z3_i32_
  %res = field.pf.sub %c0, %c1 : !PF1
  return %res : !PF1
}

// CHECK-LABEL: @test_lower_sub_vec
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]], %[[RHS:.*]]: [[T]]) -> [[T]] {
func.func @test_lower_sub_vec(%lhs : !PFv, %rhs : !PFv) -> !PFv {
  // CHECK-NOT: field.pf.sub
  // CHECK: %[[RES:.*]] = mod_arith.sub %[[LHS]], %[[RHS]] : tensor<4x!Z3_i32_>
  %res = field.pf.sub %lhs, %rhs : !PFv
  // CHECK: return %[[RES]] : [[T]]
  return %res : !PFv
}

// CHECK-LABEL: @test_lower_mul
func.func @test_lower_mul() -> !PF1 {
  // CHECK: %[[C0:.*]] = mod_arith.constant 4 : !Z3_i32_
  %c0 = field.pf.constant 4 : !PF1
  // CHECK: %[[RES:.*]] = mod_arith.mul %[[C0]], %[[C0]] : !Z3_i32_
  %res = field.pf.mul %c0, %c0 : !PF1
  return %res : !PF1
}

// CHECK-LABEL: @test_lower_mul_vec
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]], %[[RHS:.*]]: [[T]]) -> [[T]] {
func.func @test_lower_mul_vec(%lhs : !PFv, %rhs : !PFv) -> !PFv {
  // CHECK-NOT: field.pf.mul
  // CHECK: %[[RES:.*]] = mod_arith.mul %[[LHS]], %[[RHS]] : tensor<4x!Z3_i32_>
  %res = field.pf.mul %lhs, %rhs : !PFv
  // CHECK: return %[[RES]] : [[T]]
  return %res : !PFv
}
