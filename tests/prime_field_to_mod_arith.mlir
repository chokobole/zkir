// RUN: tools/zkir-opt -prime-field-to-mod-arith --split-input-file %s | FileCheck %s --enable-var-scope
!PF1 = !field.pf<3:i32>

// CHECK-LABEL: @test_lower_constant
func.func @test_lower_constant() {
  // CHECK: %[[RES:.*]] = mod_arith.constant 4 : !Z3_i32_
  %res = field.pf.constant 4 : !PF1
  return
}
