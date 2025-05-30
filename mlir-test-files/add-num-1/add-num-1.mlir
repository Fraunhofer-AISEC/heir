func.func @func(
  %arg0: tensor<8xi16>,   %arg1: tensor<8xi16>
) -> tensor<8xi16> {
  //-------------------------------------------------------------------------
  // Level 1: 0 additions (0 squarings)
  %m1 = arith.muli %arg0, %arg0 : tensor<8xi16>

  //-------------------------------------------------------------------------
  // Level 2: 1 additions (1 squaring)
  %s2_0 = arith.muli %arg0, %arg0 : tensor<8xi16>
  %l2_1 = arith.addi %m1, %s2_0 : tensor<8xi16>
  %m2 = arith.muli %l2_1, %l2_1 : tensor<8xi16>

  //-------------------------------------------------------------------------
  // Level 3: 1 additions (2 squarings)
  %s3_0_1 = arith.muli %arg0, %arg0 : tensor<8xi16>
  %s3_0_2 = arith.muli %s3_0_1, %s3_0_1 : tensor<8xi16>
  %l3_1 = arith.addi %m2, %s3_0_2 : tensor<8xi16>
  %m3 = arith.muli %l3_1, %l3_1 : tensor<8xi16>

  //-------------------------------------------------------------------------
  // Level 4: 1 additions (3 squarings)
  %s4_0_1 = arith.muli %arg0, %arg0 : tensor<8xi16>
  %s4_0_2 = arith.muli %s4_0_1, %s4_0_1 : tensor<8xi16>
  %s4_0_3 = arith.muli %s4_0_2, %s4_0_2 : tensor<8xi16>
  %l4_1 = arith.addi %m3, %s4_0_3 : tensor<8xi16>
  %m4 = arith.muli %l4_1, %l4_1 : tensor<8xi16>

  //-------------------------------------------------------------------------
  // Level 5: 1 additions (4 squarings)
  %s5_0_1 = arith.muli %arg0, %arg0 : tensor<8xi16>
  %s5_0_2 = arith.muli %s5_0_1, %s5_0_1 : tensor<8xi16>
  %s5_0_3 = arith.muli %s5_0_2, %s5_0_2 : tensor<8xi16>
  %s5_0_4 = arith.muli %s5_0_3, %s5_0_3 : tensor<8xi16>
  %l5_1 = arith.addi %m4, %s5_0_4 : tensor<8xi16>

  return %l5_1 : tensor<8xi16>
}