// RUN: zkir-opt -field-to-mod-arith %s | FileCheck %s -enable-var-scope

!PF = !field.pf<7:i32, true>

// CHECK-LABEL: @test_affine_for
// CHECK-SAME: (%[[INPUT:.*]]: [[INPUT_TYPE:.*]], %[[OUTPUT:.*]]: [[OUTPUT_TYPE:.*]]) -> [[T:.*]] {
func.func @test_affine_for(%input : memref<4x!PF>, %output : memref<4x!PF>) -> memref<4x!PF> {
  // CHECK: affine.for %[[I:.*]] = 0 to 4 {
  // CHECK:   %[[LOAD:.*]] = affine.load %[[INPUT]][%[[I]]] : [[INPUT_TYPE]]
  // CHECK:   %[[DOUBLE:.*]] = mod_arith.double %[[LOAD]] : [[MOD_ARITH_TYPE:.*]]
  // CHECK:   affine.store %[[DOUBLE]], %[[OUTPUT]][%[[I]]] : [[OUTPUT_TYPE]]
  // CHECK: }
  affine.for %i = 0 to 4 {
    %load = affine.load %input[%i] : memref<4x!PF>
    %double = field.double %load : !PF
    affine.store %double, %output[%i] : memref<4x!PF>
  }
  // CHECK: return %[[OUTPUT]] : [[T]]
  return %output : memref<4x!PF>
}

// CHECK-LABEL: @test_affine_parallel
// CHECK-SAME: (%[[INPUT:.*]]: [[INPUT_TYPE:.*]], %[[OUTPUT:.*]]: [[OUTPUT_TYPE:.*]]) -> [[T:.*]] {
func.func @test_affine_parallel(%input : memref<4x!PF>, %output : memref<4x!PF>) -> memref<4x!PF> {
  // CHECK: affine.parallel (%[[I:.*]]) = (0) to (4) {
  // CHECK:   %[[LOAD:.*]] = affine.load %[[INPUT]][%[[I]]] : [[INPUT_TYPE]]
  // CHECK:   %[[DOUBLE:.*]] = mod_arith.add %[[LOAD]], %[[LOAD]] : [[MOD_ARITH_TYPE:.*]]
  // CHECK:   affine.store %[[DOUBLE]], %[[OUTPUT]][%[[I]]] : [[OUTPUT_TYPE]]
  // CHECK: }
  affine.parallel (%i) = (0) to (4) {
    %load = affine.load %input[%i] : memref<4x!PF>
    %double = field.add %load, %load : !PF
    affine.store %double, %output[%i] : memref<4x!PF>
  }
  // CHECK: return %[[OUTPUT]] : [[T]]
  return %output : memref<4x!PF>
}
