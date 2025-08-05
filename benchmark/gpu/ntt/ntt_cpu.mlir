!PF = !field.pf<9223372036836950017 : i64>
!PFm = !field.pf<9223372036836950017 : i64, true>

#root_elem = #field.pf.elem<1184485956253136200:i64> : !PF
#root_of_unity = #field.root_of_unity<#root_elem, 1048576:i64>

func.func @ntt_cpu(%arg0 : memref<1048576x!PFm>) attributes { llvm.emit_c_interface } {
  %t = bufferization.to_tensor %arg0 restrict writable : memref<1048576x!PFm> to tensor<1048576x!PFm>
  %res = poly.ntt %t into %t {root=#root_of_unity} : tensor<1048576x!PFm>
  bufferization.materialize_in_destination %res in writable %arg0 : (tensor<1048576x!PFm>, memref<1048576x!PFm>) -> ()
  return
}

func.func @intt_cpu(%arg0 : memref<1048576x!PFm>) attributes { llvm.emit_c_interface } {
  %t = bufferization.to_tensor %arg0 restrict writable : memref<1048576x!PFm> to tensor<1048576x!PFm>
  %res = poly.ntt %t into %t {root=#root_of_unity} inverse=true : tensor<1048576x!PFm>
  bufferization.materialize_in_destination %res in writable %arg0 : (tensor<1048576x!PFm>, memref<1048576x!PFm>) -> ()
  return
}
