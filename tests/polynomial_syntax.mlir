// RUN: tools/zkir-opt --split-input-file %s | FileCheck %s --enable-var-scope

!PF1 = !field.pf<7:i32>
!PF2 = !field.pf<13:i32>
!poly_ty1 = !poly.polynomial<!PF1, 32>
!poly_ty2 = !poly.polynomial<!PF2, 32>
#uni_poly = #poly.univariate_polynomial<x**6 + 1> : !poly_ty2

// CHECK-LABEL: @test_poly_syntax
func.func @test_poly_syntax() {
  // CHECK: %[[C0:.*]] = poly.constant<1 + x**3> : [[T:.*]]
  %0 = poly.constant<x**3 + 1> : !poly_ty1
  // CHECK: %[[C1:.*]] = poly.constant<2 + x**3> : [[T]]
  %1 = poly.constant<x**3 + 2> : !poly_ty1
  // CHECK: %[[RES0:.*]] = poly.add %[[C0]], %[[C1]] : [[T]]
  %2 = poly.add %0, %1 : !poly_ty1
  // CHECK: %[[RES1:.*]] = poly.sub %[[C0]], %[[C1]] : [[T]]
  %3 = poly.sub %0, %1 : !poly_ty1
  // CHECK: %[[RES2:.*]] = poly.mul %[[C0]], %[[C1]] : [[T]]
  %4 = poly.mul %0, %1 : !poly_ty1
  return
}
