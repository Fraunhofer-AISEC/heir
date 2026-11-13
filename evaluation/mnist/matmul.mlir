func.func @main(%input: tensor<9xi32> {secret.secret}, %mat: tensor<4x9xi32>) -> tensor<4xi32> {
  %cst = arith.constant dense<0> : tensor<4xi32>
  %1 = linalg.matvec ins(%mat, %input : tensor<4x9xi32>, tensor<9xi32>) outs(%cst : tensor<4xi32>) -> tensor<4xi32>
  return %1 : tensor<4xi32>
}