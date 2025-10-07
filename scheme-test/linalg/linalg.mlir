module {
  func.func @main(%A: tensor<2x2xi32>, %x: tensor<2xi32> {secret.secret}) -> tensor<2xi32> {
    %init = tensor.empty() : tensor<2xi32>
    %y = linalg.vecmat
      ins(%x, %A: tensor<2xi32>, tensor<2x2xi32>)
      outs(%init : tensor<2xi32>) -> tensor<2xi32>
    return %y : tensor<2xi32>
  }
}