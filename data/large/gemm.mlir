#upmem = #upmem.platform<type = v1A, dpus = 2048, tasklets = 16>
func.func @gemm(%A: tensor<2048x2048xi32>, %B: tensor<2048x2048xi32>) -> tensor<2048x2048xi32>
    attributes {cinm.available_platforms = [#upmem]} {
  %r = cinm.compute -> tensor<2048x2048xi32> {
    %g = cinm.op.gemm %A, %B : tensor<2048x2048xi32>, tensor<2048x2048xi32> -> tensor<2048x2048xi32>
    cinm.yield %g : tensor<2048x2048xi32>
  }
  return %r : tensor<2048x2048xi32>
}
