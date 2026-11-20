func.func @func(%arg0: tensor<8xi16>, %arg1: tensor<8xi16>, %arg2: tensor<8xi16>, %arg3: tensor<8xi16>, %arg4: tensor<8xi16>, %arg5: tensor<8xi16>, %arg6: tensor<8xi16>, %arg7: tensor<8xi16>) -> tensor<8xi16> {
  // Plaintext weights
  %c1 = arith.constant dense<1> : tensor<8xi16>
  %c2 = arith.constant dense<2> : tensor<8xi16>
  %c3 = arith.constant dense<3> : tensor<8xi16>
  %c4 = arith.constant dense<4> : tensor<8xi16>
  %c5 = arith.constant dense<5> : tensor<8xi16>
  %c6 = arith.constant dense<6> : tensor<8xi16>
  %c7 = arith.constant dense<7> : tensor<8xi16>
  %c8 = arith.constant dense<8> : tensor<8xi16>

  // Multiply each input by its weight (plaintext * ciphertext multiplications)
  %t0 = arith.muli %c1, %arg0 : tensor<8xi16>
  %t1 = arith.muli %c2, %arg1 : tensor<8xi16>
  %t2 = arith.muli %c3, %arg2 : tensor<8xi16>
  %t3 = arith.muli %c4, %arg3 : tensor<8xi16>
  %t4 = arith.muli %c5, %arg4 : tensor<8xi16>
  %t5 = arith.muli %c6, %arg5 : tensor<8xi16>
  %t6 = arith.muli %c7, %arg6 : tensor<8xi16>
  %t7 = arith.muli %c8, %arg7 : tensor<8xi16>

  // Sum all weighted inputs
  %s1 = arith.addi %t0, %t1 : tensor<8xi16>
  %s2 = arith.addi %s1, %t2 : tensor<8xi16>
  %s3 = arith.addi %s2, %t3 : tensor<8xi16>
  %s4 = arith.addi %s3, %t4 : tensor<8xi16>
  %s5 = arith.addi %s4, %t5 : tensor<8xi16>
  %s6 = arith.addi %s5, %t6 : tensor<8xi16>
  %s7 = arith.addi %s6, %t7 : tensor<8xi16>
  return %s7 : tensor<8xi16>
}
