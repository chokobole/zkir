// RUN: zkir-opt %s --mod-arith-to-arith -convert-elementwise-to-linalg --one-shot-bufferize --convert-scf-to-cf --convert-cf-to-llvm --convert-to-llvm --convert-vector-to-llvm \
// RUN:   | mlir-runner -e test_lower_inverse -entry-point-result=void \
// RUN:      --shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t
// RUN: FileCheck %s --check-prefix=CHECK_TEST_INVERSE < %t

// RUN: zkir-opt %s --mod-arith-to-arith -convert-elementwise-to-linalg --one-shot-bufferize --convert-scf-to-cf --convert-cf-to-llvm --convert-to-llvm --convert-vector-to-llvm \
// RUN:   | mlir-runner -e test_lower_mont_reduce -entry-point-result=void \
// RUN:      --shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t
// RUN: FileCheck %s --check-prefix=CHECK_TEST_MONT_REDUCE < %t

!Fr = !mod_arith.int<2147483647:i32>

func.func private @printMemrefI32(memref<*xi32>) attributes { llvm.emit_c_interface }

func.func @test_lower_inverse() {
  %p = mod_arith.constant 3723 : !Fr
  %1 = mod_arith.inverse %p : !Fr
  %2 = mod_arith.extract %1 : !Fr -> i32
  %3 = tensor.from_elements %2 : tensor<1xi32>

  %4 = bufferization.to_memref %3 : tensor<1xi32> to memref<1xi32>
  %U = memref.cast %4 : memref<1xi32> to memref<*xi32>
  func.call @printMemrefI32(%U) : (memref<*xi32>) -> ()
  return
}

// CHECK_TEST_INVERSE: [1324944920]

!Fq = !mod_arith.int<21888242871839275222246405745257275088548364400416034343698204186575808495617 : i256>
#Fq_mont = #mod_arith.montgomery<!Fq>

func.func @test_lower_mont_reduce() {
  %p = arith.constant 3723 : i512
  %p_mont = mod_arith.mont_reduce %p {montgomery=#Fq_mont} : i512 -> !Fq

  %2 = mod_arith.extract %p_mont : !Fq -> i256
  %3 = vector.from_elements %2 : vector<1xi256>
  %4 = vector.bitcast %3 : vector<1xi256> to vector<8xi32>
  %mem = memref.alloc() : memref<8xi32>
  %idx_0 = arith.constant 0 : index
  vector.store %4, %mem[%idx_0] : memref<8xi32>, vector<8xi32>

  %U = memref.cast %mem : memref<8xi32> to memref<*xi32>
  func.call @printMemrefI32(%U) : (memref<*xi32>) -> ()
  return
}

// CHECK_TEST_MONT_REDUCE: [-1635059004, -1772563805, -2074116324, -156049350, 156881531, -524227392, -1359481138, 438709201]
