func.func @func(%arg0: tensor<8xi16>, %arg1: tensor<8xi16>, %arg2: tensor<8xi16>, %arg3: tensor<8xi16>) -> tensor<8xi16> {
  // Plaintext matrix entries (1..16)
  %c1 = arith.constant dense<1> : tensor<8xi16>
  %c2 = arith.constant dense<2> : tensor<8xi16>
  %c3 = arith.constant dense<3> : tensor<8xi16>
  %c4 = arith.constant dense<4> : tensor<8xi16>
  %c5 = arith.constant dense<5> : tensor<8xi16>
  %c6 = arith.constant dense<6> : tensor<8xi16>
  %c7 = arith.constant dense<7> : tensor<8xi16>
  %c8 = arith.constant dense<8> : tensor<8xi16>
  %c9 = arith.constant dense<9> : tensor<8xi16>
  %c10 = arith.constant dense<10> : tensor<8xi16>
  %c11 = arith.constant dense<11> : tensor<8xi16>
  %c12 = arith.constant dense<12> : tensor<8xi16>
  %c13 = arith.constant dense<13> : tensor<8xi16>
  %c14 = arith.constant dense<14> : tensor<8xi16>
  %c15 = arith.constant dense<15> : tensor<8xi16>
  %c16 = arith.constant dense<16> : tensor<8xi16>

  // 16 plaintext * ciphertext multiplications
  %m00 = arith.muli %c1, %arg0 : tensor<8xi16>
  %m01 = arith.muli %c2, %arg1 : tensor<8xi16>
  %m02 = arith.muli %c3, %arg2 : tensor<8xi16>
  %m03 = arith.muli %c4, %arg3 : tensor<8xi16>
  %m10 = arith.muli %c5, %arg0 : tensor<8xi16>
  %m11 = arith.muli %c6, %arg1 : tensor<8xi16>
  %m12 = arith.muli %c7, %arg2 : tensor<8xi16>
  %m13 = arith.muli %c8, %arg3 : tensor<8xi16>
  %m20 = arith.muli %c9, %arg0 : tensor<8xi16>
  %m21 = arith.muli %c10, %arg1 : tensor<8xi16>
  %m22 = arith.muli %c11, %arg2 : tensor<8xi16>
  %m23 = arith.muli %c12, %arg3 : tensor<8xi16>
  %m30 = arith.muli %c13, %arg0 : tensor<8xi16>
  %m31 = arith.muli %c14, %arg1 : tensor<8xi16>
  %m32 = arith.muli %c15, %arg2 : tensor<8xi16>
  %m33 = arith.muli %c16, %arg3 : tensor<8xi16>

  // Sum all 16 products
  %a0 = arith.addi %m00, %m01 : tensor<8xi16>
  %a1 = arith.addi %a0, %m02 : tensor<8xi16>
  %a2 = arith.addi %a1, %m03 : tensor<8xi16>
  %a3 = arith.addi %a2, %m10 : tensor<8xi16>
  %a4 = arith.addi %a3, %m11 : tensor<8xi16>
  %a5 = arith.addi %a4, %m12 : tensor<8xi16>
  %a6 = arith.addi %a5, %m13 : tensor<8xi16>
  %a7 = arith.addi %a6, %m20 : tensor<8xi16>
  %a8 = arith.addi %a7, %m21 : tensor<8xi16>
  %a9 = arith.addi %a8, %m22 : tensor<8xi16>
  %a10 = arith.addi %a9, %m23 : tensor<8xi16>
  %a11 = arith.addi %a10, %m30 : tensor<8xi16>
  %a12 = arith.addi %a11, %m31 : tensor<8xi16>
  %a13 = arith.addi %a12, %m32 : tensor<8xi16>
  %a14 = arith.addi %a13, %m33 : tensor<8xi16>

  return %a14 : tensor<8xi16>
}
