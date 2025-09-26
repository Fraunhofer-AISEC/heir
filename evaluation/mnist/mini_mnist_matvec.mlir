module {
  func.func @main(%input: tensor<25xi32>,
                  %layer1_weights: tensor<15x25xi32>,
                  %layer1_bias: tensor<15xi32>) -> tensor<15xi32> {
    %c_0_i32 = arith.constant dense<0> : tensor<15xi32>

    // Matvec für erste Schicht (Matrix x Vektor)
    %layer1_mul = linalg.matvec ins(%layer1_weights, %input : tensor<15x25xi32>, tensor<25xi32>) outs(%c_0_i32 : tensor<15xi32>) -> tensor<15xi32>

    // Bias hinzufügen
    %layer1_with_bias = arith.addi %layer1_mul, %layer1_bias : tensor<15xi32>

    // ReLU anwenden: max(0, x)
    //%relu_mask = arith.cmpi sgt, %layer1_with_bias, %c_0_i32 : tensor<15xi32>
    //%layer1_out = arith.select %relu_mask, %layer1_with_bias, %c_0_i32 : tensor<15xi32>

    func.return %layer1_with_bias : tensor<15xi32>  }
}