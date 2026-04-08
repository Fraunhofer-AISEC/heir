func.func @func(
  %arg0: tensor<8xi16>,  %arg1: tensor<8xi16>,  %arg2: tensor<8xi16>,  %arg3: tensor<8xi16>,
  %arg4: tensor<8xi16>,  %arg5: tensor<8xi16>,  %arg6: tensor<8xi16>,  %arg7: tensor<8xi16>,
  %arg8: tensor<8xi16>,  %arg9: tensor<8xi16>,  %arg10: tensor<8xi16>, %arg11: tensor<8xi16>,
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
  %c0 = arith.constant dense<0> : tensor<8xi16>

  // Process %arg0.
  %s5_0_1 = arith.muli %arg0, %arg0 : tensor<8xi16>
  %s5_0_2 = arith.muli %s5_0_1, %s5_0_1 : tensor<8xi16>
  %s5_0_3 = arith.muli %s5_0_2, %s5_0_2 : tensor<8xi16>
  %s5_0_4 = arith.muli %s5_0_3, %s5_0_3 : tensor<8xi16>
  %l5_1  = arith.addi %c0, %s5_0_4 : tensor<8xi16>

  // Process %arg1.
  %s5_1_1 = arith.muli %arg1, %arg1 : tensor<8xi16>
  %s5_1_2 = arith.muli %s5_1_1, %s5_1_1 : tensor<8xi16>
  %s5_1_3 = arith.muli %s5_1_2, %s5_1_2 : tensor<8xi16>
  %s5_1_4 = arith.muli %s5_1_3, %s5_1_3 : tensor<8xi16>
  %l5_2  = arith.addi %l5_1, %s5_1_4 : tensor<8xi16>

  // Process %arg2.
  %s5_2_1 = arith.muli %arg2, %arg2 : tensor<8xi16>
  %s5_2_2 = arith.muli %s5_2_1, %s5_2_1 : tensor<8xi16>
  %s5_2_3 = arith.muli %s5_2_2, %s5_2_2 : tensor<8xi16>
  %s5_2_4 = arith.muli %s5_2_3, %s5_2_3 : tensor<8xi16>
  %l5_3  = arith.addi %l5_2, %s5_2_4 : tensor<8xi16>

  // Process %arg3.
  %s5_3_1 = arith.muli %arg3, %arg3 : tensor<8xi16>
  %s5_3_2 = arith.muli %s5_3_1, %s5_3_1 : tensor<8xi16>
  %s5_3_3 = arith.muli %s5_3_2, %s5_3_2 : tensor<8xi16>
  %s5_3_4 = arith.muli %s5_3_3, %s5_3_3 : tensor<8xi16>
  %l5_4  = arith.addi %l5_3, %s5_3_4 : tensor<8xi16>

  // Process %arg4.
  %s5_4_1 = arith.muli %arg4, %arg4 : tensor<8xi16>
  %s5_4_2 = arith.muli %s5_4_1, %s5_4_1 : tensor<8xi16>
  %s5_4_3 = arith.muli %s5_4_2, %s5_4_2 : tensor<8xi16>
  %s5_4_4 = arith.muli %s5_4_3, %s5_4_3 : tensor<8xi16>
  %l5_5  = arith.addi %l5_4, %s5_4_4 : tensor<8xi16>

  // Process %arg5.
  %s5_5_1 = arith.muli %arg5, %arg5 : tensor<8xi16>
  %s5_5_2 = arith.muli %s5_5_1, %s5_5_1 : tensor<8xi16>
  %s5_5_3 = arith.muli %s5_5_2, %s5_5_2 : tensor<8xi16>
  %s5_5_4 = arith.muli %s5_5_3, %s5_5_3 : tensor<8xi16>
  %l5_6  = arith.addi %l5_5, %s5_5_4 : tensor<8xi16>

  // Process %arg6.
  %s5_6_1 = arith.muli %arg6, %arg6 : tensor<8xi16>
  %s5_6_2 = arith.muli %s5_6_1, %s5_6_1 : tensor<8xi16>
  %s5_6_3 = arith.muli %s5_6_2, %s5_6_2 : tensor<8xi16>
  %s5_6_4 = arith.muli %s5_6_3, %s5_6_3 : tensor<8xi16>
  %l5_7  = arith.addi %l5_6, %s5_6_4 : tensor<8xi16>

  // Process %arg7.
  %s5_7_1 = arith.muli %arg7, %arg7 : tensor<8xi16>
  %s5_7_2 = arith.muli %s5_7_1, %s5_7_1 : tensor<8xi16>
  %s5_7_3 = arith.muli %s5_7_2, %s5_7_2 : tensor<8xi16>
  %s5_7_4 = arith.muli %s5_7_3, %s5_7_3 : tensor<8xi16>
  %l5_8  = arith.addi %l5_7, %s5_7_4 : tensor<8xi16>

  // Process %arg8.
  %s5_8_1 = arith.muli %arg8, %arg8 : tensor<8xi16>
  %s5_8_2 = arith.muli %s5_8_1, %s5_8_1 : tensor<8xi16>
  %s5_8_3 = arith.muli %s5_8_2, %s5_8_2 : tensor<8xi16>
  %s5_8_4 = arith.muli %s5_8_3, %s5_8_3 : tensor<8xi16>
  %l5_9  = arith.addi %l5_8, %s5_8_4 : tensor<8xi16>

  // Process %arg9.
  %s5_9_1 = arith.muli %arg9, %arg9 : tensor<8xi16>
  %s5_9_2 = arith.muli %s5_9_1, %s5_9_1 : tensor<8xi16>
  %s5_9_3 = arith.muli %s5_9_2, %s5_9_2 : tensor<8xi16>
  %s5_9_4 = arith.muli %s5_9_3, %s5_9_3 : tensor<8xi16>
  %l5_10 = arith.addi %l5_9, %s5_9_4 : tensor<8xi16>

  // Process %arg10.
  %s5_10_1 = arith.muli %arg10, %arg10 : tensor<8xi16>
  %s5_10_2 = arith.muli %s5_10_1, %s5_10_1 : tensor<8xi16>
  %s5_10_3 = arith.muli %s5_10_2, %s5_10_2 : tensor<8xi16>
  %s5_10_4 = arith.muli %s5_10_3, %s5_10_3 : tensor<8xi16>
  %l5_11 = arith.addi %l5_10, %s5_10_4 : tensor<8xi16>

  // Process %arg11.
  %s5_11_1 = arith.muli %arg11, %arg11 : tensor<8xi16>
  %s5_11_2 = arith.muli %s5_11_1, %s5_11_1 : tensor<8xi16>
  %s5_11_3 = arith.muli %s5_11_2, %s5_11_2 : tensor<8xi16>
  %s5_11_4 = arith.muli %s5_11_3, %s5_11_3 : tensor<8xi16>
  %l5_12 = arith.addi %l5_11, %s5_11_4 : tensor<8xi16>

  // Process %arg12.
  %s5_12_1 = arith.muli %arg12, %arg12 : tensor<8xi16>
  %s5_12_2 = arith.muli %s5_12_1, %s5_12_1 : tensor<8xi16>
  %s5_12_3 = arith.muli %s5_12_2, %s5_12_2 : tensor<8xi16>
  %s5_12_4 = arith.muli %s5_12_3, %s5_12_3 : tensor<8xi16>
  %l5_13 = arith.addi %l5_12, %s5_12_4 : tensor<8xi16>

  // Process %arg13.
  %s5_13_1 = arith.muli %arg13, %arg13 : tensor<8xi16>
  %s5_13_2 = arith.muli %s5_13_1, %s5_13_1 : tensor<8xi16>
  %s5_13_3 = arith.muli %s5_13_2, %s5_13_2 : tensor<8xi16>
  %s5_13_4 = arith.muli %s5_13_3, %s5_13_3 : tensor<8xi16>
  %l5_14 = arith.addi %l5_13, %s5_13_4 : tensor<8xi16>

  // Process %arg14.
  %s5_14_1 = arith.muli %arg14, %arg14 : tensor<8xi16>
  %s5_14_2 = arith.muli %s5_14_1, %s5_14_1 : tensor<8xi16>
  %s5_14_3 = arith.muli %s5_14_2, %s5_14_2 : tensor<8xi16>
  %s5_14_4 = arith.muli %s5_14_3, %s5_14_3 : tensor<8xi16>
  %l5_15 = arith.addi %l5_14, %s5_14_4 : tensor<8xi16>

  // Process %arg15.
  %s5_15_1 = arith.muli %arg15, %arg15 : tensor<8xi16>
  %s5_15_2 = arith.muli %s5_15_1, %s5_15_1 : tensor<8xi16>
  %s5_15_3 = arith.muli %s5_15_2, %s5_15_2 : tensor<8xi16>
  %s5_15_4 = arith.muli %s5_15_3, %s5_15_3 : tensor<8xi16>
  %l5_16 = arith.addi %l5_15, %s5_15_4 : tensor<8xi16>

  // Process %arg16.
  %s5_16_1 = arith.muli %arg16, %arg16 : tensor<8xi16>
  %s5_16_2 = arith.muli %s5_16_1, %s5_16_1 : tensor<8xi16>
  %s5_16_3 = arith.muli %s5_16_2, %s5_16_2 : tensor<8xi16>
  %s5_16_4 = arith.muli %s5_16_3, %s5_16_3 : tensor<8xi16>
  %l5_17 = arith.addi %l5_16, %s5_16_4 : tensor<8xi16>

  // Process %arg17.
  %s5_17_1 = arith.muli %arg17, %arg17 : tensor<8xi16>
  %s5_17_2 = arith.muli %s5_17_1, %s5_17_1 : tensor<8xi16>
  %s5_17_3 = arith.muli %s5_17_2, %s5_17_2 : tensor<8xi16>
  %s5_17_4 = arith.muli %s5_17_3, %s5_17_3 : tensor<8xi16>
  %l5_18 = arith.addi %l5_17, %s5_17_4 : tensor<8xi16>

  // Process %arg18.
  %s5_18_1 = arith.muli %arg18, %arg18 : tensor<8xi16>
  %s5_18_2 = arith.muli %s5_18_1, %s5_18_1 : tensor<8xi16>
  %s5_18_3 = arith.muli %s5_18_2, %s5_18_2 : tensor<8xi16>
  %s5_18_4 = arith.muli %s5_18_3, %s5_18_3 : tensor<8xi16>
  %l5_19 = arith.addi %l5_18, %s5_18_4 : tensor<8xi16>

  // Process %arg19.
  %s5_19_1 = arith.muli %arg19, %arg19 : tensor<8xi16>
  %s5_19_2 = arith.muli %s5_19_1, %s5_19_1 : tensor<8xi16>
  %s5_19_3 = arith.muli %s5_19_2, %s5_19_2 : tensor<8xi16>
  %s5_19_4 = arith.muli %s5_19_3, %s5_19_3 : tensor<8xi16>
  %l5_20 = arith.addi %l5_19, %s5_19_4 : tensor<8xi16>

  // Process %arg20.
  %s5_20_1 = arith.muli %arg20, %arg20 : tensor<8xi16>
  %s5_20_2 = arith.muli %s5_20_1, %s5_20_1 : tensor<8xi16>
  %s5_20_3 = arith.muli %s5_20_2, %s5_20_2 : tensor<8xi16>
  %s5_20_4 = arith.muli %s5_20_3, %s5_20_3 : tensor<8xi16>
  %l5_21 = arith.addi %l5_20, %s5_20_4 : tensor<8xi16>

  // Process %arg21.
  %s5_21_1 = arith.muli %arg21, %arg21 : tensor<8xi16>
  %s5_21_2 = arith.muli %s5_21_1, %s5_21_1 : tensor<8xi16>
  %s5_21_3 = arith.muli %s5_21_2, %s5_21_2 : tensor<8xi16>
  %s5_21_4 = arith.muli %s5_21_3, %s5_21_3 : tensor<8xi16>
  %l5_22 = arith.addi %l5_21, %s5_21_4 : tensor<8xi16>

  // Process %arg22.
  %s5_22_1 = arith.muli %arg22, %arg22 : tensor<8xi16>
  %s5_22_2 = arith.muli %s5_22_1, %s5_22_1 : tensor<8xi16>
  %s5_22_3 = arith.muli %s5_22_2, %s5_22_2 : tensor<8xi16>
  %s5_22_4 = arith.muli %s5_22_3, %s5_22_3 : tensor<8xi16>
  %l5_23 = arith.addi %l5_22, %s5_22_4 : tensor<8xi16>

  // Process %arg23.
  %s5_23_1 = arith.muli %arg23, %arg23 : tensor<8xi16>
  %s5_23_2 = arith.muli %s5_23_1, %s5_23_1 : tensor<8xi16>
  %s5_23_3 = arith.muli %s5_23_2, %s5_23_2 : tensor<8xi16>
  %s5_23_4 = arith.muli %s5_23_3, %s5_23_3 : tensor<8xi16>
  %l5_24 = arith.addi %l5_23, %s5_23_4 : tensor<8xi16>

  // Process %arg24.
  %s5_24_1 = arith.muli %arg24, %arg24 : tensor<8xi16>
  %s5_24_2 = arith.muli %s5_24_1, %s5_24_1 : tensor<8xi16>
  %s5_24_3 = arith.muli %s5_24_2, %s5_24_2 : tensor<8xi16>
  %s5_24_4 = arith.muli %s5_24_3, %s5_24_3 : tensor<8xi16>
  %l5_25 = arith.addi %l5_24, %s5_24_4 : tensor<8xi16>

  // Process %arg25.
  %s5_25_1 = arith.muli %arg25, %arg25 : tensor<8xi16>
  %s5_25_2 = arith.muli %s5_25_1, %s5_25_1 : tensor<8xi16>
  %s5_25_3 = arith.muli %s5_25_2, %s5_25_2 : tensor<8xi16>
  %s5_25_4 = arith.muli %s5_25_3, %s5_25_3 : tensor<8xi16>
  %l5_26 = arith.addi %l5_25, %s5_25_4 : tensor<8xi16>

  // Process %arg26.
  %s5_26_1 = arith.muli %arg26, %arg26 : tensor<8xi16>
  %s5_26_2 = arith.muli %s5_26_1, %s5_26_1 : tensor<8xi16>
  %s5_26_3 = arith.muli %s5_26_2, %s5_26_2 : tensor<8xi16>
  %s5_26_4 = arith.muli %s5_26_3, %s5_26_3 : tensor<8xi16>
  %l5_27 = arith.addi %l5_26, %s5_26_4 : tensor<8xi16>

  // Process %arg27.
  %s5_27_1 = arith.muli %arg27, %arg27 : tensor<8xi16>
  %s5_27_2 = arith.muli %s5_27_1, %s5_27_1 : tensor<8xi16>
  %s5_27_3 = arith.muli %s5_27_2, %s5_27_2 : tensor<8xi16>
  %s5_27_4 = arith.muli %s5_27_3, %s5_27_3 : tensor<8xi16>
  %l5_28 = arith.addi %l5_27, %s5_27_4 : tensor<8xi16>

  // Process %arg28.
  %s5_28_1 = arith.muli %arg28, %arg28 : tensor<8xi16>
  %s5_28_2 = arith.muli %s5_28_1, %s5_28_1 : tensor<8xi16>
  %s5_28_3 = arith.muli %s5_28_2, %s5_28_2 : tensor<8xi16>
  %s5_28_4 = arith.muli %s5_28_3, %s5_28_3 : tensor<8xi16>
  %l5_29 = arith.addi %l5_28, %s5_28_4 : tensor<8xi16>

  // Process %arg29.
  %s5_29_1 = arith.muli %arg29, %arg29 : tensor<8xi16>
  %s5_29_2 = arith.muli %s5_29_1, %s5_29_1 : tensor<8xi16>
  %s5_29_3 = arith.muli %s5_29_2, %s5_29_2 : tensor<8xi16>
  %s5_29_4 = arith.muli %s5_29_3, %s5_29_3 : tensor<8xi16>
  %l5_30 = arith.addi %l5_29, %s5_29_4 : tensor<8xi16>

  // Process %arg30.
  %s5_30_1 = arith.muli %arg30, %arg30 : tensor<8xi16>
  %s5_30_2 = arith.muli %s5_30_1, %s5_30_1 : tensor<8xi16>
  %s5_30_3 = arith.muli %s5_30_2, %s5_30_2 : tensor<8xi16>
  %s5_30_4 = arith.muli %s5_30_3, %s5_30_3 : tensor<8xi16>
  %l5_31 = arith.addi %l5_30, %s5_30_4 : tensor<8xi16>

  // Process %arg31.
  %s5_31_1 = arith.muli %arg31, %arg31 : tensor<8xi16>
  %s5_31_2 = arith.muli %s5_31_1, %s5_31_1 : tensor<8xi16>
  %s5_31_3 = arith.muli %s5_31_2, %s5_31_2 : tensor<8xi16>
  %s5_31_4 = arith.muli %s5_31_3, %s5_31_3 : tensor<8xi16>
  %l5_32 = arith.addi %l5_31, %s5_31_4 : tensor<8xi16>

  // Process %arg32.
  %s5_32_1 = arith.muli %arg32, %arg32 : tensor<8xi16>
  %s5_32_2 = arith.muli %s5_32_1, %s5_32_1 : tensor<8xi16>
  %s5_32_3 = arith.muli %s5_32_2, %s5_32_2 : tensor<8xi16>
  %s5_32_4 = arith.muli %s5_32_3, %s5_32_3 : tensor<8xi16>
  %l5_33 = arith.addi %l5_32, %s5_32_4 : tensor<8xi16>

  // Process %arg33.
  %s5_33_1 = arith.muli %arg33, %arg33 : tensor<8xi16>
  %s5_33_2 = arith.muli %s5_33_1, %s5_33_1 : tensor<8xi16>
  %s5_33_3 = arith.muli %s5_33_2, %s5_33_2 : tensor<8xi16>
  %s5_33_4 = arith.muli %s5_33_3, %s5_33_3 : tensor<8xi16>
  %l5_34 = arith.addi %l5_33, %s5_33_4 : tensor<8xi16>

  // Process %arg34.
  %s5_34_1 = arith.muli %arg34, %arg34 : tensor<8xi16>
  %s5_34_2 = arith.muli %s5_34_1, %s5_34_1 : tensor<8xi16>
  %s5_34_3 = arith.muli %s5_34_2, %s5_34_2 : tensor<8xi16>
  %s5_34_4 = arith.muli %s5_34_3, %s5_34_3 : tensor<8xi16>
  %l5_35 = arith.addi %l5_33, %s5_34_4 : tensor<8xi16>

  // Process %arg35.
  %s5_35_1 = arith.muli %arg35, %arg35 : tensor<8xi16>
  %s5_35_2 = arith.muli %s5_35_1, %s5_35_1 : tensor<8xi16>
  %s5_35_3 = arith.muli %s5_35_2, %s5_35_2 : tensor<8xi16>
  %s5_35_4 = arith.muli %s5_35_3, %s5_35_3 : tensor<8xi16>
  %l5_36 = arith.addi %l5_35, %s5_35_4 : tensor<8xi16>

  // Process %arg36.
  %s5_36_1 = arith.muli %arg36, %arg36 : tensor<8xi16>
  %s5_36_2 = arith.muli %s5_36_1, %s5_36_1 : tensor<8xi16>
  %s5_36_3 = arith.muli %s5_36_2, %s5_36_2 : tensor<8xi16>
  %s5_36_4 = arith.muli %s5_36_3, %s5_36_3 : tensor<8xi16>
  %l5_37 = arith.addi %l5_36, %s5_36_4 : tensor<8xi16>

  // Process %arg37.
  %s5_37_1 = arith.muli %arg37, %arg37 : tensor<8xi16>
  %s5_37_2 = arith.muli %s5_37_1, %s5_37_1 : tensor<8xi16>
  %s5_37_3 = arith.muli %s5_37_2, %s5_37_2 : tensor<8xi16>
  %s5_37_4 = arith.muli %s5_37_3, %s5_37_3 : tensor<8xi16>
  %l5_38 = arith.addi %l5_37, %s5_37_4 : tensor<8xi16>

  // Process %arg38.
  %s5_38_1 = arith.muli %arg38, %arg38 : tensor<8xi16>
  %s5_38_2 = arith.muli %s5_38_1, %s5_38_1 : tensor<8xi16>
  %s5_38_3 = arith.muli %s5_38_2, %s5_38_2 : tensor<8xi16>
  %s5_38_4 = arith.muli %s5_38_3, %s5_38_3 : tensor<8xi16>
  %l5_39 = arith.addi %l5_38, %s5_38_4 : tensor<8xi16>

  // Process %arg39.
  %s5_39_1 = arith.muli %arg39, %arg39 : tensor<8xi16>
  %s5_39_2 = arith.muli %s5_39_1, %s5_39_1 : tensor<8xi16>
  %s5_39_3 = arith.muli %s5_39_2, %s5_39_2 : tensor<8xi16>
  %s5_39_4 = arith.muli %s5_39_3, %s5_39_3 : tensor<8xi16>
  %l5_40 = arith.addi %l5_39, %s5_39_4 : tensor<8xi16>

  // Process %arg40.
  %s5_40_1 = arith.muli %arg40, %arg40 : tensor<8xi16>
  %s5_40_2 = arith.muli %s5_40_1, %s5_40_1 : tensor<8xi16>
  %s5_40_3 = arith.muli %s5_40_2, %s5_40_2 : tensor<8xi16>
  %s5_40_4 = arith.muli %s5_40_3, %s5_40_3 : tensor<8xi16>
  %l5_41 = arith.addi %l5_40, %s5_40_4 : tensor<8xi16>

  // Process %arg41.
  %s5_41_1 = arith.muli %arg41, %arg41 : tensor<8xi16>
  %s5_41_2 = arith.muli %s5_41_1, %s5_41_1 : tensor<8xi16>
  %s5_41_3 = arith.muli %s5_41_2, %s5_41_2 : tensor<8xi16>
  %s5_41_4 = arith.muli %s5_41_3, %s5_41_3 : tensor<8xi16>
  %l5_42 = arith.addi %l5_41, %s5_41_4 : tensor<8xi16>

  // Process %arg42.
  %s5_42_1 = arith.muli %arg42, %arg42 : tensor<8xi16>
  %s5_42_2 = arith.muli %s5_42_1, %s5_42_1 : tensor<8xi16>
  %s5_42_3 = arith.muli %s5_42_2, %s5_42_2 : tensor<8xi16>
  %s5_42_4 = arith.muli %s5_42_3, %s5_42_3 : tensor<8xi16>
  %l5_43 = arith.addi %l5_42, %s5_42_4 : tensor<8xi16>

  // Process %arg43.
  %s5_43_1 = arith.muli %arg43, %arg43 : tensor<8xi16>
  %s5_43_2 = arith.muli %s5_43_1, %s5_43_1 : tensor<8xi16>
  %s5_43_3 = arith.muli %s5_43_2, %s5_43_2 : tensor<8xi16>
  %s5_43_4 = arith.muli %s5_43_3, %s5_43_3 : tensor<8xi16>
  %l5_44 = arith.addi %l5_43, %s5_43_4 : tensor<8xi16>

  // Process %arg44.
  %s5_44_1 = arith.muli %arg44, %arg44 : tensor<8xi16>
  %s5_44_2 = arith.muli %s5_44_1, %s5_44_1 : tensor<8xi16>
  %s5_44_3 = arith.muli %s5_44_2, %s5_44_2 : tensor<8xi16>
  %s5_44_4 = arith.muli %s5_44_3, %s5_44_3 : tensor<8xi16>
  %l5_45 = arith.addi %l5_44, %s5_44_4 : tensor<8xi16>

  // Process %arg45.
  %s5_45_1 = arith.muli %arg45, %arg45 : tensor<8xi16>
  %s5_45_2 = arith.muli %s5_45_1, %s5_45_1 : tensor<8xi16>
  %s5_45_3 = arith.muli %s5_45_2, %s5_45_2 : tensor<8xi16>
  %s5_45_4 = arith.muli %s5_45_3, %s5_45_3 : tensor<8xi16>
  %l5_46 = arith.addi %l5_44, %s5_45_4 : tensor<8xi16>

  // Process %arg46.
  %s5_46_1 = arith.muli %arg46, %arg46 : tensor<8xi16>
  %s5_46_2 = arith.muli %s5_46_1, %s5_46_1 : tensor<8xi16>
  %s5_46_3 = arith.muli %s5_46_2, %s5_46_2 : tensor<8xi16>
  %s5_46_4 = arith.muli %s5_46_3, %s5_46_3 : tensor<8xi16>
  %l5_47 = arith.addi %l5_46, %s5_46_4 : tensor<8xi16>

  // Process %arg47.
  %s5_47_1 = arith.muli %arg47, %arg47 : tensor<8xi16>
  %s5_47_2 = arith.muli %s5_47_1, %s5_47_1 : tensor<8xi16>
  %s5_47_3 = arith.muli %s5_47_2, %s5_47_2 : tensor<8xi16>
  %s5_47_4 = arith.muli %s5_47_3, %s5_47_3 : tensor<8xi16>
  %l5_48 = arith.addi %l5_47, %s5_47_4 : tensor<8xi16>

  // Process %arg48.
  %s5_48_1 = arith.muli %arg48, %arg48 : tensor<8xi16>
  %s5_48_2 = arith.muli %s5_48_1, %s5_48_1 : tensor<8xi16>
  %s5_48_3 = arith.muli %s5_48_2, %s5_48_2 : tensor<8xi16>
  %s5_48_4 = arith.muli %s5_48_3, %s5_48_3 : tensor<8xi16>
  %l5_49 = arith.addi %l5_48, %s5_48_4 : tensor<8xi16>

  // Process %arg49.
  %s5_49_1 = arith.muli %arg49, %arg49 : tensor<8xi16>
  %s5_49_2 = arith.muli %s5_49_1, %s5_49_1 : tensor<8xi16>
  %s5_49_3 = arith.muli %s5_49_2, %s5_49_2 : tensor<8xi16>
  %s5_49_4 = arith.muli %s5_49_3, %s5_49_3 : tensor<8xi16>
  %l5_50 = arith.addi %l5_49, %s5_49_4 : tensor<8xi16>

  // Process %arg50.
  %s5_50_1 = arith.muli %arg50, %arg50 : tensor<8xi16>
  %s5_50_2 = arith.muli %s5_50_1, %s5_50_1 : tensor<8xi16>
  %s5_50_3 = arith.muli %s5_50_2, %s5_50_2 : tensor<8xi16>
  %s5_50_4 = arith.muli %s5_50_3, %s5_50_3 : tensor<8xi16>
  %l5_51 = arith.addi %l5_50, %s5_50_4 : tensor<8xi16>

  // Process %arg51.
  %s5_51_1 = arith.muli %arg51, %arg51 : tensor<8xi16>
  %s5_51_2 = arith.muli %s5_51_1, %s5_51_1 : tensor<8xi16>
  %s5_51_3 = arith.muli %s5_51_2, %s5_51_2 : tensor<8xi16>
  %s5_51_4 = arith.muli %s5_51_3, %s5_51_3 : tensor<8xi16>
  %l5_52 = arith.addi %l5_51, %s5_51_4 : tensor<8xi16>

  // Process %arg52.
  %s5_52_1 = arith.muli %arg52, %arg52 : tensor<8xi16>
  %s5_52_2 = arith.muli %s5_52_1, %s5_52_1 : tensor<8xi16>
  %s5_52_3 = arith.muli %s5_52_2, %s5_52_2 : tensor<8xi16>
  %s5_52_4 = arith.muli %s5_52_3, %s5_52_3 : tensor<8xi16>
  %l5_53 = arith.addi %l5_52, %s5_52_4 : tensor<8xi16>

  // Process %arg53.
  %s5_53_1 = arith.muli %arg53, %arg53 : tensor<8xi16>
  %s5_53_2 = arith.muli %s5_53_1, %s5_53_1 : tensor<8xi16>
  %s5_53_3 = arith.muli %s5_53_2, %s5_53_2 : tensor<8xi16>
  %s5_53_4 = arith.muli %s5_53_3, %s5_53_3 : tensor<8xi16>
  %l5_54 = arith.addi %l5_53, %s5_53_4 : tensor<8xi16>

  // Process %arg54.
  %s5_54_1 = arith.muli %arg54, %arg54 : tensor<8xi16>
  %s5_54_2 = arith.muli %s5_54_1, %s5_54_1 : tensor<8xi16>
  %s5_54_3 = arith.muli %s5_54_2, %s5_54_2 : tensor<8xi16>
  %s5_54_4 = arith.muli %s5_54_3, %s5_54_3 : tensor<8xi16>
  %l5_55 = arith.addi %l5_54, %s5_54_4 : tensor<8xi16>

  // Process %arg55.
  %s5_55_1 = arith.muli %arg55, %arg55 : tensor<8xi16>
  %s5_55_2 = arith.muli %s5_55_1, %s5_55_1 : tensor<8xi16>
  %s5_55_3 = arith.muli %s5_55_2, %s5_55_2 : tensor<8xi16>
  %s5_55_4 = arith.muli %s5_55_3, %s5_55_3 : tensor<8xi16>
  %l5_56 = arith.addi %l5_55, %s5_55_4 : tensor<8xi16>

  // Process %arg56.
  %s5_56_1 = arith.muli %arg56, %arg56 : tensor<8xi16>
  %s5_56_2 = arith.muli %s5_56_1, %s5_56_1 : tensor<8xi16>
  %s5_56_3 = arith.muli %s5_56_2, %s5_56_2 : tensor<8xi16>
  %s5_56_4 = arith.muli %s5_56_3, %s5_56_3 : tensor<8xi16>
  %l5_57 = arith.addi %l5_56, %s5_56_4 : tensor<8xi16>

  // Process %arg57.
  %s5_57_1 = arith.muli %arg57, %arg57 : tensor<8xi16>
  %s5_57_2 = arith.muli %s5_57_1, %s5_57_1 : tensor<8xi16>
  %s5_57_3 = arith.muli %s5_57_2, %s5_57_2 : tensor<8xi16>
  %s5_57_4 = arith.muli %s5_57_3, %s5_57_3 : tensor<8xi16>
  %l5_58 = arith.addi %l5_57, %s5_57_4 : tensor<8xi16>

  // Process %arg58.
  %s5_58_1 = arith.muli %arg58, %arg58 : tensor<8xi16>
  %s5_58_2 = arith.muli %s5_58_1, %s5_58_1 : tensor<8xi16>
  %s5_58_3 = arith.muli %s5_58_2, %s5_58_2 : tensor<8xi16>
  %s5_58_4 = arith.muli %s5_58_3, %s5_58_3 : tensor<8xi16>
  %l5_59 = arith.addi %l5_58, %s5_58_4 : tensor<8xi16>

  // Process %arg59.
  %s5_59_1 = arith.muli %arg59, %arg59 : tensor<8xi16>
  %s5_59_2 = arith.muli %s5_59_1, %s5_59_1 : tensor<8xi16>
  %s5_59_3 = arith.muli %s5_59_2, %s5_59_2 : tensor<8xi16>
  %s5_59_4 = arith.muli %s5_59_3, %s5_59_3 : tensor<8xi16>
  %l5_60 = arith.addi %l5_59, %s5_59_4 : tensor<8xi16>

  // Process %arg60.
  %s5_60_1 = arith.muli %arg60, %arg60 : tensor<8xi16>
  %s5_60_2 = arith.muli %s5_60_1, %s5_60_1 : tensor<8xi16>
  %s5_60_3 = arith.muli %s5_60_2, %s5_60_2 : tensor<8xi16>
  %s5_60_4 = arith.muli %s5_60_3, %s5_60_3 : tensor<8xi16>
  %l5_61 = arith.addi %l5_60, %s5_60_4 : tensor<8xi16>

  // Process %arg61.
  %s5_61_1 = arith.muli %arg61, %arg61 : tensor<8xi16>
  %s5_61_2 = arith.muli %s5_61_1, %s5_61_1 : tensor<8xi16>
  %s5_61_3 = arith.muli %s5_61_2, %s5_61_2 : tensor<8xi16>
  %s5_61_4 = arith.muli %s5_61_3, %s5_61_3 : tensor<8xi16>
  %l5_62 = arith.addi %l5_61, %s5_61_4 : tensor<8xi16>

  // Process %arg62.
  %s5_62_1 = arith.muli %arg62, %arg62 : tensor<8xi16>
  %s5_62_2 = arith.muli %s5_62_1, %s5_62_1 : tensor<8xi16>
  %s5_62_3 = arith.muli %s5_62_2, %s5_62_2 : tensor<8xi16>
  %s5_62_4 = arith.muli %s5_62_3, %s5_62_3 : tensor<8xi16>
  %l5_63 = arith.addi %l5_62, %s5_62_4 : tensor<8xi16>

  // Process %arg63.
  %s5_63_1 = arith.muli %arg63, %arg63 : tensor<8xi16>
  %s5_63_2 = arith.muli %s5_63_1, %s5_63_1 : tensor<8xi16>
  %s5_63_3 = arith.muli %s5_63_2, %s5_63_2 : tensor<8xi16>
  %s5_63_4 = arith.muli %s5_63_3, %s5_63_3 : tensor<8xi16>
  %l5_64 = arith.addi %l5_63, %s5_63_4 : tensor<8xi16>
  
  %l5_65 = arith.addi %l5_64, %s5_0_4 : tensor<8xi16>
  %l5_66 = arith.addi %l5_65, %s5_1_4 : tensor<8xi16>
  %l5_67 = arith.addi %l5_66, %s5_2_4 : tensor<8xi16>
  %l5_68 = arith.addi %l5_67, %s5_3_4 : tensor<8xi16>
  %l5_69 = arith.addi %l5_68, %s5_4_4 : tensor<8xi16>
  %l5_70 = arith.addi %l5_69, %s5_5_4 : tensor<8xi16>
  %l5_71 = arith.addi %l5_70, %s5_6_4 : tensor<8xi16>
  %l5_72 = arith.addi %l5_71, %s5_7_4 : tensor<8xi16>
  %l5_73 = arith.addi %l5_72, %s5_8_4 : tensor<8xi16>
  %l5_74 = arith.addi %l5_73, %s5_9_4 : tensor<8xi16>
  %l5_75 = arith.addi %l5_74, %s5_10_4 : tensor<8xi16>
  %l5_76 = arith.addi %l5_75, %s5_11_4 : tensor<8xi16>
  %l5_77 = arith.addi %l5_76, %s5_12_4 : tensor<8xi16>
  %l5_78 = arith.addi %l5_77, %s5_13_4 : tensor<8xi16>
  %l5_79 = arith.addi %l5_78, %s5_14_4 : tensor<8xi16>
  %l5_80 = arith.addi %l5_79, %s5_15_4 : tensor<8xi16>
  %l5_81 = arith.addi %l5_80, %s5_16_4 : tensor<8xi16>
  %l5_82 = arith.addi %l5_81, %s5_17_4 : tensor<8xi16>
  %l5_83 = arith.addi %l5_82, %l5_82 : tensor<8xi16>
  %l5_84 = arith.addi %l5_83, %s5_18_4 : tensor<8xi16>

  return %l5_84 : tensor<8xi16>
}
