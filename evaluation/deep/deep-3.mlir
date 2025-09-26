func.func @main(%a: tensor<2xi32>, %b: tensor<2xi32>) -> tensor<2xi32> {
    %result = arith.addi %a, %b : tensor<2xi32>
    return %result : tensor<2xi32>
}