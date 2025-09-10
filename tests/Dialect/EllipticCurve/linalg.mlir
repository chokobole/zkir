// RUN: cat %S/../../bn254_field_defs.mlir %S/../../bn254_ec_mont_defs.mlir %s \
// RUN:   | zkir-opt -linalg-generalize-named-ops -split-input-file \
// RUN:   | FileCheck %s -enable-var-scope

// CHECK-LABEL: @test_g2_msm_by_dot_product
func.func @test_g2_msm_by_dot_product(%scalars: tensor<3x!PF>, %points: tensor<3x!g2jacobian>) -> tensor<!g2jacobian> {
  // CHECK-NOT: linalg.dot
  %result = tensor.empty() : tensor<!g2jacobian>
  %msm_result = linalg.dot ins(%scalars, %points : tensor<3x!PF>, tensor<3x!g2jacobian>) outs(%result: tensor<!g2jacobian>) -> tensor<!g2jacobian>
  return %msm_result : tensor<!g2jacobian>
}
