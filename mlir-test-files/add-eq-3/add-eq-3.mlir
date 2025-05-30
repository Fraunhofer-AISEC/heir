func.func @func(
  %arg0: tensor<8xi16>, %arg1: tensor<8xi16>, %arg2: tensor<8xi16>, %arg3: tensor<8xi16>,
  %arg4: tensor<8xi16>, %arg5: tensor<8xi16>, %arg6: tensor<8xi16>, %arg7: tensor<8xi16>,
  %arg8: tensor<8xi16>, %arg9: tensor<8xi16>, %arg10: tensor<8xi16>, %arg11: tensor<8xi16>,
  %arg12: tensor<8xi16>, %arg13: tensor<8xi16>, %arg14: tensor<8xi16>, %arg15: tensor<8xi16>,
  %arg16: tensor<8xi16>, %arg17: tensor<8xi16>, %arg18: tensor<8xi16>, %arg19: tensor<8xi16>,
  %arg20: tensor<8xi16>, %arg21: tensor<8xi16>, %arg22: tensor<8xi16>, %arg23: tensor<8xi16>,
  %arg24: tensor<8xi16>, %arg25: tensor<8xi16>, %arg26: tensor<8xi16>, %arg27: tensor<8xi16>,
  %arg28: tensor<8xi16>, %arg29: tensor<8xi16>, %arg30: tensor<8xi16>, %arg31: tensor<8xi16>,
  %arg32: tensor<8xi16>, %arg33: tensor<8xi16>, %arg34: tensor<8xi16>, %arg35: tensor<8xi16>,
  %arg36: tensor<8xi16>, %arg37: tensor<8xi16>, %arg38: tensor<8xi16>, %arg39: tensor<8xi16>,
  %arg40: tensor<8xi16>, %arg41: tensor<8xi16>, %arg42: tensor<8xi16>, %arg43: tensor<8xi16>,
  %arg44: tensor<8xi16>, %arg45: tensor<8xi16>, %arg46: tensor<8xi16>, %arg47: tensor<8xi16>,
  %arg48: tensor<8xi16>, %arg49: tensor<8xi16>, %arg50: tensor<8xi16>, %arg51: tensor<8xi16>,
  %arg52: tensor<8xi16>, %arg53: tensor<8xi16>, %arg54: tensor<8xi16>, %arg55: tensor<8xi16>,
  %arg56: tensor<8xi16>, %arg57: tensor<8xi16>, %arg58: tensor<8xi16>, %arg59: tensor<8xi16>,
  %arg60: tensor<8xi16>, %arg61: tensor<8xi16>, %arg62: tensor<8xi16>, %arg63: tensor<8xi16>
) -> tensor<8xi16> {
  //-------------------------------------------------------------------------
  // Level 1: 0 additions (0 squarings)
  // Use operand: %arg0 (no additions, just squaring)
  %l1_sq = arith.muli %arg0, %arg0 : tensor<8xi16>
  //-------------------------------------------------------------------------
  // Level 2: 32 additions (1 squaring)
  // Use operands: %arg0 .. %arg31
  %s2_0  = arith.muli %arg0, %arg0 : tensor<8xi16>
  %l2_1  = arith.addi %l1_sq, %s2_0 : tensor<8xi16>
  %s2_1  = arith.muli %arg1, %arg1 : tensor<8xi16>
  %l2_2  = arith.addi %l2_1, %s2_1 : tensor<8xi16>
  %s2_2  = arith.muli %arg2, %arg2 : tensor<8xi16>
  %l2_3  = arith.addi %l2_2, %s2_2 : tensor<8xi16>
  %s2_3  = arith.muli %arg3, %arg3 : tensor<8xi16>
  %l2_4  = arith.addi %l2_3, %s2_3 : tensor<8xi16>
  %s2_4  = arith.muli %arg4, %arg4 : tensor<8xi16>
  %l2_5  = arith.addi %l2_4, %s2_4 : tensor<8xi16>
  %s2_5  = arith.muli %arg5, %arg5 : tensor<8xi16>
  %l2_6  = arith.addi %l2_5, %s2_5 : tensor<8xi16>
  %s2_6  = arith.muli %arg6, %arg6 : tensor<8xi16>
  %l2_7  = arith.addi %l2_6, %s2_6 : tensor<8xi16>
  %s2_7  = arith.muli %arg7, %arg7 : tensor<8xi16>
  %l2_8  = arith.addi %l2_7, %s2_7 : tensor<8xi16>
  %s2_8  = arith.muli %arg8, %arg8 : tensor<8xi16>
  %l2_9  = arith.addi %l2_8, %s2_8 : tensor<8xi16>
  %s2_9  = arith.muli %arg9, %arg9 : tensor<8xi16>
  %l2_10 = arith.addi %l2_9, %s2_9 : tensor<8xi16>
  %s2_10  = arith.muli %arg10, %arg10 : tensor<8xi16>
  %l2_11 = arith.addi %l2_10, %s2_10 : tensor<8xi16>
  %s2_11  = arith.muli %arg11, %arg11 : tensor<8xi16>
  %l2_12 = arith.addi %l2_11, %s2_11 : tensor<8xi16>
  %s2_12  = arith.muli %arg12, %arg12 : tensor<8xi16>
  %l2_13 = arith.addi %l2_12, %s2_12 : tensor<8xi16>
  %s2_13  = arith.muli %arg13, %arg13 : tensor<8xi16>
  %l2_14 = arith.addi %l2_13, %s2_13 : tensor<8xi16>
  %s2_14  = arith.muli %arg14, %arg14 : tensor<8xi16>
  %l2_15 = arith.addi %l2_14, %s2_14 : tensor<8xi16>
  %s2_15  = arith.muli %arg15, %arg15 : tensor<8xi16>
  %l2_16 = arith.addi %l2_15, %s2_15 : tensor<8xi16>
  %s2_16  = arith.muli %arg16, %arg16 : tensor<8xi16>
  %l2_17 = arith.addi %l2_16, %s2_16 : tensor<8xi16>
  %s2_17  = arith.muli %arg17, %arg17 : tensor<8xi16>
  %l2_18 = arith.addi %l2_17, %s2_17 : tensor<8xi16>
  %s2_18  = arith.muli %arg18, %arg18 : tensor<8xi16>
  %l2_19 = arith.addi %l2_18, %s2_18 : tensor<8xi16>
  %s2_19  = arith.muli %arg19, %arg19 : tensor<8xi16>
  %l2_20 = arith.addi %l2_19, %s2_19 : tensor<8xi16>
  %s2_20  = arith.muli %arg20, %arg20 : tensor<8xi16>
  %l2_21 = arith.addi %l2_20, %s2_20 : tensor<8xi16>
  %s2_21  = arith.muli %arg21, %arg21 : tensor<8xi16>
  %l2_22 = arith.addi %l2_21, %s2_21 : tensor<8xi16>
  %s2_22  = arith.muli %arg22, %arg22 : tensor<8xi16>
  %l2_23 = arith.addi %l2_22, %s2_22 : tensor<8xi16>
  %s2_23  = arith.muli %arg23, %arg23 : tensor<8xi16>
  %l2_24 = arith.addi %l2_23, %s2_23 : tensor<8xi16>
  %s2_24  = arith.muli %arg24, %arg24 : tensor<8xi16>
  %l2_25 = arith.addi %l2_24, %s2_24 : tensor<8xi16>
  %s2_25  = arith.muli %arg25, %arg25 : tensor<8xi16>
  %l2_26 = arith.addi %l2_25, %s2_25 : tensor<8xi16>
  %s2_26  = arith.muli %arg26, %arg26 : tensor<8xi16>
  %l2_27 = arith.addi %l2_26, %s2_26 : tensor<8xi16>
  %s2_27  = arith.muli %arg27, %arg27 : tensor<8xi16>
  %l2_28 = arith.addi %l2_27, %s2_27 : tensor<8xi16>
  %s2_28  = arith.muli %arg28, %arg28 : tensor<8xi16>
  %l2_29 = arith.addi %l2_28, %s2_28 : tensor<8xi16>
  %s2_29  = arith.muli %arg29, %arg29 : tensor<8xi16>
  %l2_30 = arith.addi %l2_29, %s2_29 : tensor<8xi16>
  %s2_30  = arith.muli %arg30, %arg30 : tensor<8xi16>
  %l2_31 = arith.addi %l2_30, %s2_30 : tensor<8xi16>
  %s2_31  = arith.muli %arg31, %arg31 : tensor<8xi16>
  %l2_32 = arith.addi %l2_31, %s2_31 : tensor<8xi16>
  %l2_sq = arith.muli %l2_32, %l2_32 : tensor<8xi16>
  //-------------------------------------------------------------------------
  // Level 3: 32 additions (2 squarings)
  // Use operands: %arg0..%arg31
  %s3_0_2 = arith.muli %s2_0, %s2_0 : tensor<8xi16>
  %l3_1   = arith.addi %l2_sq, %s3_0_2 : tensor<8xi16>
  %s3_1_2 = arith.muli %s2_1, %s2_1 : tensor<8xi16>
  %l3_2   = arith.addi %l3_1, %s3_1_2 : tensor<8xi16>
  %s3_2_2 = arith.muli %s2_2, %s2_2 : tensor<8xi16>
  %l3_3   = arith.addi %l3_2, %s3_2_2 : tensor<8xi16>
  %s3_3_2 = arith.muli %s2_3, %s2_3 : tensor<8xi16>
  %l3_4   = arith.addi %l3_3, %s3_3_2 : tensor<8xi16>
  %s3_4_2 = arith.muli %s2_4, %s2_4 : tensor<8xi16>
  %l3_5   = arith.addi %l3_4, %s3_4_2 : tensor<8xi16>
  %s3_5_2 = arith.muli %s2_5, %s2_5 : tensor<8xi16>
  %l3_6   = arith.addi %l3_5, %s3_5_2 : tensor<8xi16>
  %s3_6_2 = arith.muli %s2_6, %s2_6 : tensor<8xi16>
  %l3_7   = arith.addi %l3_6, %s3_6_2 : tensor<8xi16>
  %s3_7_2 = arith.muli %s2_7, %s2_7 : tensor<8xi16>
  %l3_8   = arith.addi %l3_7, %s3_7_2 : tensor<8xi16>
  %s3_8_2 = arith.muli %s2_8, %s2_8 : tensor<8xi16>
  %l3_9   = arith.addi %l3_8, %s3_8_2 : tensor<8xi16>
  %s3_9_2 = arith.muli %s2_9, %s2_9 : tensor<8xi16>
  %l3_10  = arith.addi %l3_9, %s3_9_2 : tensor<8xi16>
  %s3_10_2 = arith.muli %s2_10, %s2_10 : tensor<8xi16>
  %l3_11  = arith.addi %l3_10, %s3_10_2 : tensor<8xi16>
  %s3_11_2 = arith.muli %s2_11, %s2_11 : tensor<8xi16>
  %l3_12  = arith.addi %l3_11, %s3_11_2 : tensor<8xi16>
  %s3_12_2 = arith.muli %s2_12, %s2_12 : tensor<8xi16>
  %l3_13  = arith.addi %l3_12, %s3_12_2 : tensor<8xi16>
  %s3_13_2 = arith.muli %s2_13, %s2_13 : tensor<8xi16>
  %l3_14  = arith.addi %l3_13, %s3_13_2 : tensor<8xi16>
  %s3_14_2 = arith.muli %s2_14, %s2_14 : tensor<8xi16>
  %l3_15  = arith.addi %l3_14, %s3_14_2 : tensor<8xi16>
  %s3_15_2 = arith.muli %s2_15, %s2_15 : tensor<8xi16>
  %l3_16  = arith.addi %l3_15, %s3_15_2 : tensor<8xi16>
  %s3_16_2 = arith.muli %s2_16, %s2_16 : tensor<8xi16>
  %l3_17  = arith.addi %l3_16, %s3_16_2 : tensor<8xi16>
  %s3_17_2 = arith.muli %s2_17, %s2_17 : tensor<8xi16>
  %l3_18  = arith.addi %l3_17, %s3_17_2 : tensor<8xi16>
  %s3_18_2 = arith.muli %s2_18, %s2_18 : tensor<8xi16>
  %l3_19  = arith.addi %l3_18, %s3_18_2 : tensor<8xi16>
  %s3_19_2 = arith.muli %s2_19, %s2_19 : tensor<8xi16>
  %l3_20  = arith.addi %l3_19, %s3_19_2 : tensor<8xi16>
  %s3_20_2 = arith.muli %s2_20, %s2_20 : tensor<8xi16>
  %l3_21  = arith.addi %l3_20, %s3_20_2 : tensor<8xi16>
  %s3_21_2 = arith.muli %s2_21, %s2_21 : tensor<8xi16>
  %l3_22  = arith.addi %l3_21, %s3_21_2 : tensor<8xi16>
  %s3_22_2 = arith.muli %s2_22, %s2_22 : tensor<8xi16>
  %l3_23  = arith.addi %l3_22, %s3_22_2 : tensor<8xi16>
  %s3_23_2 = arith.muli %s2_23, %s2_23 : tensor<8xi16>
  %l3_24  = arith.addi %l3_23, %s3_23_2 : tensor<8xi16>
  %s3_24_2 = arith.muli %s2_24, %s2_24 : tensor<8xi16>
  %l3_25  = arith.addi %l3_24, %s3_24_2 : tensor<8xi16>
  %s3_25_2 = arith.muli %s2_25, %s2_25 : tensor<8xi16>
  %l3_26  = arith.addi %l3_25, %s3_25_2 : tensor<8xi16>
  %s3_26_2 = arith.muli %s2_26, %s2_26 : tensor<8xi16>
  %l3_27  = arith.addi %l3_26, %s3_26_2 : tensor<8xi16>
  %s3_27_2 = arith.muli %s2_27, %s2_27 : tensor<8xi16>
  %l3_28  = arith.addi %l3_27, %s3_27_2 : tensor<8xi16>
  %s3_28_2 = arith.muli %s2_28, %s2_28 : tensor<8xi16>
  %l3_29  = arith.addi %l3_28, %s3_28_2 : tensor<8xi16>
  %s3_29_2 = arith.muli %s2_29, %s2_29 : tensor<8xi16>
  %l3_30  = arith.addi %l3_29, %s3_29_2 : tensor<8xi16>
  %s3_30_2 = arith.muli %s2_30, %s2_30 : tensor<8xi16>
  %l3_31  = arith.addi %l3_30, %s3_30_2 : tensor<8xi16>
  %s3_31_2 = arith.muli %s2_31, %s2_31 : tensor<8xi16>
  %l3_32  = arith.addi %l3_31, %s3_31_2 : tensor<8xi16>
  %l3_sq = arith.muli %l3_32, %l3_32 : tensor<8xi16>
  //-------------------------------------------------------------------------
  // Level 4: 32 additions (3 squarings)
  // Use operands: %arg0..%arg31
  %s4_0_3 = arith.muli %s3_0_2, %s3_0_2 : tensor<8xi16>
  %l4_1   = arith.addi %l3_sq, %s4_0_3 : tensor<8xi16>
  %s4_1_3 = arith.muli %s3_1_2, %s3_1_2 : tensor<8xi16>
  %l4_2   = arith.addi %l4_1, %s4_1_3 : tensor<8xi16>
  %s4_2_3 = arith.muli %s3_2_2, %s3_2_2 : tensor<8xi16>
  %l4_3   = arith.addi %l4_2, %s4_2_3 : tensor<8xi16>
  %s4_3_3 = arith.muli %s3_3_2, %s3_3_2 : tensor<8xi16>
  %l4_4   = arith.addi %l4_3, %s4_3_3 : tensor<8xi16>
  %s4_4_3 = arith.muli %s3_4_2, %s3_4_2 : tensor<8xi16>
  %l4_5   = arith.addi %l4_4, %s4_4_3 : tensor<8xi16>
  %s4_5_3 = arith.muli %s3_5_2, %s3_5_2 : tensor<8xi16>
  %l4_6   = arith.addi %l4_5, %s4_5_3 : tensor<8xi16>
  %s4_6_3 = arith.muli %s3_6_2, %s3_6_2 : tensor<8xi16>
  %l4_7   = arith.addi %l4_6, %s4_6_3 : tensor<8xi16>
  %s4_7_3 = arith.muli %s3_7_2, %s3_7_2 : tensor<8xi16>
  %l4_8   = arith.addi %l4_7, %s4_7_3 : tensor<8xi16>
  %s4_8_3 = arith.muli %s3_8_2, %s3_8_2 : tensor<8xi16>
  %l4_9   = arith.addi %l4_8, %s4_8_3 : tensor<8xi16>
  %s4_9_3 = arith.muli %s3_9_2, %s3_9_2 : tensor<8xi16>
  %l4_10  = arith.addi %l4_9, %s4_9_3 : tensor<8xi16>
  %s4_10_3 = arith.muli %s3_10_2, %s3_10_2 : tensor<8xi16>
  %l4_11  = arith.addi %l4_10, %s4_10_3 : tensor<8xi16>
  %s4_11_3 = arith.muli %s3_11_2, %s3_11_2 : tensor<8xi16>
  %l4_12  = arith.addi %l4_11, %s4_11_3 : tensor<8xi16>
  %s4_12_3 = arith.muli %s3_12_2, %s3_12_2 : tensor<8xi16>
  %l4_13  = arith.addi %l4_12, %s4_12_3 : tensor<8xi16>
  %s4_13_3 = arith.muli %s3_13_2, %s3_13_2 : tensor<8xi16>
  %l4_14  = arith.addi %l4_13, %s4_13_3 : tensor<8xi16>
  %s4_14_3 = arith.muli %s3_14_2, %s3_14_2 : tensor<8xi16>
  %l4_15  = arith.addi %l4_14, %s4_14_3 : tensor<8xi16>
  %s4_15_3 = arith.muli %s3_15_2, %s3_15_2 : tensor<8xi16>
  %l4_16  = arith.addi %l4_15, %s4_15_3 : tensor<8xi16>
  %s4_16_3 = arith.muli %s3_16_2, %s3_16_2 : tensor<8xi16>
  %l4_17  = arith.addi %l4_16, %s4_16_3 : tensor<8xi16>
  %s4_17_3 = arith.muli %s3_17_2, %s3_17_2 : tensor<8xi16>
  %l4_18  = arith.addi %l4_17, %s4_17_3 : tensor<8xi16>
  %s4_18_3 = arith.muli %s3_18_2, %s3_18_2 : tensor<8xi16>
  %l4_19  = arith.addi %l4_18, %s4_18_3 : tensor<8xi16>
  %s4_19_3 = arith.muli %s3_19_2, %s3_19_2 : tensor<8xi16>
  %l4_20  = arith.addi %l4_19, %s4_19_3 : tensor<8xi16>
  %s4_20_3 = arith.muli %s3_20_2, %s3_20_2 : tensor<8xi16>
  %l4_21  = arith.addi %l4_20, %s4_20_3 : tensor<8xi16>
  %s4_21_3 = arith.muli %s3_21_2, %s3_21_2 : tensor<8xi16>
  %l4_22  = arith.addi %l4_21, %s4_21_3 : tensor<8xi16>
  %s4_22_3 = arith.muli %s3_22_2, %s3_22_2 : tensor<8xi16>
  %l4_23  = arith.addi %l4_22, %s4_22_3 : tensor<8xi16>
  %s4_23_3 = arith.muli %s3_23_2, %s3_23_2 : tensor<8xi16>
  %l4_24  = arith.addi %l4_23, %s4_23_3 : tensor<8xi16>
  %s4_24_3 = arith.muli %s3_24_2, %s3_24_2 : tensor<8xi16>
  %l4_25  = arith.addi %l4_24, %s4_24_3 : tensor<8xi16>
  %s4_25_3 = arith.muli %s3_25_2, %s3_25_2 : tensor<8xi16>
  %l4_26  = arith.addi %l4_25, %s4_25_3 : tensor<8xi16>
  %s4_26_3 = arith.muli %s3_26_2, %s3_26_2 : tensor<8xi16>
  %l4_27  = arith.addi %l4_26, %s4_26_3 : tensor<8xi16>
  %s4_27_3 = arith.muli %s3_27_2, %s3_27_2 : tensor<8xi16>
  %l4_28  = arith.addi %l4_27, %s4_27_3 : tensor<8xi16>
  %s4_28_3 = arith.muli %s3_28_2, %s3_28_2 : tensor<8xi16>
  %l4_29  = arith.addi %l4_28, %s4_28_3 : tensor<8xi16>
  %s4_29_3 = arith.muli %s3_29_2, %s3_29_2 : tensor<8xi16>
  %l4_30  = arith.addi %l4_29, %s4_29_3 : tensor<8xi16>
  %s4_30_3 = arith.muli %s3_30_2, %s3_30_2 : tensor<8xi16>
  %l4_31  = arith.addi %l4_30, %s4_30_3 : tensor<8xi16>
  %s4_31_3 = arith.muli %s3_31_2, %s3_31_2 : tensor<8xi16>
  %l4_32  = arith.addi %l4_31, %s4_31_3 : tensor<8xi16>
  %l4_sq = arith.muli %l4_32, %l4_32 : tensor<8xi16>
  return %l4_32 : tensor<8xi16>
}
