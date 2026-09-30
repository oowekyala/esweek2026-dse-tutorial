#upmem = #upmem.platform<type = v1A, dpus = 2048, tasklets = 16>
func.func @gemv(%A: tensor<4096x4096xi32>, %x: tensor<4096xi32>) -> tensor<4096xi32>
    attributes {cinm.available_platforms = [#upmem]} {
  %r = cinm.compute -> tensor<4096xi32> {
    %g = cinm.op.gemv %A, %x : tensor<4096x4096xi32>, tensor<4096xi32> -> tensor<4096xi32>
    cinm.yield %g : tensor<4096xi32>
  }
  return %r : tensor<4096xi32>
}
