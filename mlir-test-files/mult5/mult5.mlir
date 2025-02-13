module {
  func.func @func(%v0: tensor<4xi32>, %v1: tensor<4xi32>) -> tensor<4xi32> {
    // Multiply the two inputs (elementwise) to start the chain.
    %res1 = arith.muli %v0, %v1 : tensor<4xi32>
    // Multiply the result by the first input.
    %res2 = arith.muli %res1, %v0 : tensor<4xi32>
    // Multiply the result by the second input.
    %res3 = arith.muli %res2, %v1 : tensor<4xi32>
    // Multiply the result by the first input.
    %res4 = arith.muli %res3, %v0 : tensor<4xi32>
    // Multiply the result by the second input.
    //%res5 = arith.muli %res4, %v1 : tensor<4xi32>
    return %res4 : tensor<4xi32>
  }
}
