// RUN: zkir-opt %s -specialize-arith-to-avx -convert-to-llvm \
// RUN:   | mlir-runner -e packed_mului_extended -entry-point-result=void \
// RUN:      -shared-libs="%mlir_lib_dir/libmlir_runner_utils%shlibext" > %t
// RUN: FileCheck %s -check-prefix=CHECK_PACKED_MULUI_EXTENDED < %t

func.func private @printMemrefI32(memref<*xi32>) attributes { llvm.emit_c_interface }

func.func @packed_mului_extended() {
  %a = arith.constant dense<[1,10,100,1000,10000,100000,100000,10000000,1,10,100,1000,10000,100000,100000,10000000]> : vector<16xi32>
  %b = arith.constant dense<[1,10,100,1000,10000,100000,100000,10000000,1,10,100,1000,10000,100000,100000,10000000]> : vector<16xi32>
  %c:2 = arith.mului_extended %a, %b : vector<16xi32>

  %mem = memref.alloc() : memref<32xi32>
  %idx_low = arith.constant 0 : index
  %idx_high = arith.constant 16 : index
  vector.store %c#0, %mem[%idx_low] : memref<32xi32>, vector<16xi32>
  vector.store %c#1, %mem[%idx_high] : memref<32xi32>, vector<16xi32>

  %U = memref.cast %mem : memref<32xi32> to memref<*xi32>
  func.call @printMemrefI32(%U) : (memref<*xi32>) -> ()
  return
}

// CHECK_PACKED_MULUI_EXTENDED: [1, 100, 10000, 1000000, 100000000, 1410065408, 1410065408, 276447232, 1, 100, 10000, 1000000, 100000000, 1410065408, 1410065408, 276447232, 0, 0, 0, 0, 0, 2, 2, 23283, 0, 0, 0, 0, 0, 2, 2, 23283]
