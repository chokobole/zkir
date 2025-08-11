!PF = !field.pf<9223372036836950017 : i64>

func.func @matvec_gpu(%arg0: memref<1048576x100x!PF>, %arg1: memref<100x!PF>, %arg2: memref<1048576x!PF>) attributes {llvm.emit_c_interface}  {
  linalg.matvec ins(%arg0, %arg1: memref<1048576x100x!PF>, memref<100x!PF>) outs(%arg2: memref<1048576x!PF>)
  return
}
