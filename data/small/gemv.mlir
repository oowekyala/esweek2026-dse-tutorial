#upmem = #upmem.platform<type = v1A, dpus = 64, tasklets = 16>
func.func @gemv(%A: tensor<1024x1024xi32>, %x: tensor<1024xi32>) -> tensor<1024xi32>
    attributes {cinm.available_platforms = [#upmem]} {
  %r = cinm.compute -> tensor<1024xi32> {
    %g = cinm.op.gemv %A, %x : tensor<1024x1024xi32>, tensor<1024xi32> -> tensor<1024xi32>
    cinm.yield %g : tensor<1024xi32>
  }
  return %r : tensor<1024xi32>
}
