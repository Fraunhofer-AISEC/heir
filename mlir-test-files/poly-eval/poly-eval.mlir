func.func @func(%arg0: tensor<8xi16>) -> tensor<8xi16> {
  // Plaintext coefficients
  %c1 = arith.constant dense<1> : tensor<8xi16>
  %c2 = arith.constant dense<2> : tensor<8xi16>
  %c3 = arith.constant dense<3> : tensor<8xi16>
  %c5 = arith.constant dense<5> : tensor<8xi16>
  %c7 = arith.constant dense<7> : tensor<8xi16>

  // Compute powers of x (ciphertext * ciphertext multiplications)
  %x2 = arith.muli %arg0, %arg0 : tensor<8xi16>
  %x3 = arith.muli %x2, %arg0 : tensor<8xi16>
  %x4 = arith.muli %x3, %arg0 : tensor<8xi16>

  // Multiply coefficients by powers of x (plaintext * ciphertext multiplications)
  %t1 = arith.muli %c2, %arg0 : tensor<8xi16>
  %t2 = arith.muli %c3, %x2 : tensor<8xi16>
  %t3 = arith.muli %c5, %x3 : tensor<8xi16>
  %t4 = arith.muli %c7, %x4 : tensor<8xi16>

  // Sum: f(x) = 1 + 2x + 3x^2 + 5x^3 + 7x^4
  %r1 = arith.addi %c1, %t1 : tensor<8xi16>
  %r2 = arith.addi %r1, %t2 : tensor<8xi16>
  %r3 = arith.addi %r2, %t3 : tensor<8xi16>
  %r4 = arith.addi %r3, %t4 : tensor<8xi16>

  return %r4 : tensor<8xi16>
}
