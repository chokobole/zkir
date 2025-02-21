// RUN: tools/zkir-opt -prime-field-to-mod-arith --split-input-file %s | FileCheck %s --enable-var-scope
!PF1 = !field.pf<3:i32>

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

// CHECK-LABEL: @test_lower_mul
func.func @test_lower_mul() -> !PF1 {
  // CHECK: %[[C0:.*]] = mod_arith.constant 4 : !Z3_i32_
  %c0 = field.pf.constant 4 : !PF1
  // CHECK: %[[RES:.*]] = mod_arith.mul %[[C0]], %[[C0]] : !Z3_i32_
  %res = field.pf.mul %c0, %c0 : !PF1
  return %res : !PF1
}
