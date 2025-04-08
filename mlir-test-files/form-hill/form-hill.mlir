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
  //===-------------------------------------------------------------------===
  // Level 1: 4 additions (operands: %arg0 .. %arg4)
  // No squaring for new operands.
  %l1_1 = arith.addi %arg0, %arg1 : tensor<8xi16>
  %l1_2 = arith.addi %l1_1, %arg2 : tensor<8xi16>
  %l1_3 = arith.addi %l1_2, %arg3 : tensor<8xi16>
  %l1_4 = arith.addi %l1_3, %arg4 : tensor<8xi16>
  // Multiply the Level 1 result (barrier)
  %m1 = arith.muli %l1_4, %l1_4 : tensor<8xi16>

  //===-------------------------------------------------------------------===
  // Level 2: 16 additions (operands: %arg0 .. %arg15, each squared once)
  %s2_0  = arith.muli %arg0,  %arg0  : tensor<8xi16>
  %l2_1  = arith.addi %m1, %s2_0  : tensor<8xi16>
  %s2_1  = arith.muli %arg1,  %arg1  : tensor<8xi16>
  %l2_2  = arith.addi %l2_1, %s2_1  : tensor<8xi16>
  %s2_2  = arith.muli %arg2,  %arg2  : tensor<8xi16>
  %l2_3  = arith.addi %l2_2, %s2_2  : tensor<8xi16>
  %s2_3  = arith.muli %arg3,  %arg3  : tensor<8xi16>
  %l2_4  = arith.addi %l2_3, %s2_3  : tensor<8xi16>
  %s2_4  = arith.muli %arg4,  %arg4  : tensor<8xi16>
  %l2_5  = arith.addi %l2_4, %s2_4  : tensor<8xi16>
  %s2_5  = arith.muli %arg5,  %arg5  : tensor<8xi16>
  %l2_6  = arith.addi %l2_5, %s2_5  : tensor<8xi16>
  %s2_6  = arith.muli %arg6,  %arg6  : tensor<8xi16>
  %l2_7  = arith.addi %l2_6, %s2_6  : tensor<8xi16>
  %s2_7  = arith.muli %arg7,  %arg7  : tensor<8xi16>
  %l2_8  = arith.addi %l2_7, %s2_7  : tensor<8xi16>
  %s2_8  = arith.muli %arg8,  %arg8  : tensor<8xi16>
  %l2_9  = arith.addi %l2_8, %s2_8  : tensor<8xi16>
  %s2_9  = arith.muli %arg9,  %arg9  : tensor<8xi16>
  %l2_10 = arith.addi %l2_9, %s2_9  : tensor<8xi16>
  %s2_10 = arith.muli %arg10, %arg10 : tensor<8xi16>
  %l2_11 = arith.addi %l2_10, %s2_10 : tensor<8xi16>
  %s2_11 = arith.muli %arg11, %arg11 : tensor<8xi16>
  %l2_12 = arith.addi %l2_11, %s2_11 : tensor<8xi16>
  %s2_12 = arith.muli %arg12, %arg12 : tensor<8xi16>
  %l2_13 = arith.addi %l2_12, %s2_12 : tensor<8xi16>
  %s2_13 = arith.muli %arg13, %arg13 : tensor<8xi16>
  %l2_14 = arith.addi %l2_13, %s2_13 : tensor<8xi16>
  %s2_14 = arith.muli %arg14, %arg14 : tensor<8xi16>
  %l2_15 = arith.addi %l2_14, %s2_14 : tensor<8xi16>
  %s2_15 = arith.muli %arg15, %arg15 : tensor<8xi16>
  %l2_16 = arith.addi %l2_15, %s2_15 : tensor<8xi16>
  %m2 = arith.muli %l2_16, %l2_16 : tensor<8xi16>

  //===-------------------------------------------------------------------===
  // Level 3: 64 additions (operands: %arg0 .. %arg63, each squared twice)
  %s3_0_1 = arith.muli %arg0,  %arg0  : tensor<8xi16>
  %s3_0_2 = arith.muli %s3_0_1, %s3_0_1 : tensor<8xi16>
  %l3_1   = arith.addi %m2, %s3_0_2 : tensor<8xi16>
  %s3_1_1 = arith.muli %arg1,  %arg1  : tensor<8xi16>
  %s3_1_2 = arith.muli %s3_1_1, %s3_1_1 : tensor<8xi16>
  %l3_2   = arith.addi %l3_1, %s3_1_2 : tensor<8xi16>
  %s3_2_1 = arith.muli %arg2,  %arg2  : tensor<8xi16>
  %s3_2_2 = arith.muli %s3_2_1, %s3_2_1 : tensor<8xi16>
  %l3_3   = arith.addi %l3_2, %s3_2_2 : tensor<8xi16>
  %s3_3_1 = arith.muli %arg3,  %arg3  : tensor<8xi16>
  %s3_3_2 = arith.muli %s3_3_1, %s3_3_1 : tensor<8xi16>
  %l3_4   = arith.addi %l3_3, %s3_3_2 : tensor<8xi16>
  %s3_4_1 = arith.muli %arg4,  %arg4  : tensor<8xi16>
  %s3_4_2 = arith.muli %s3_4_1, %s3_4_1 : tensor<8xi16>
  %l3_5   = arith.addi %l3_4, %s3_4_2 : tensor<8xi16>
  %s3_5_1 = arith.muli %arg5,  %arg5  : tensor<8xi16>
  %s3_5_2 = arith.muli %s3_5_1, %s3_5_1 : tensor<8xi16>
  %l3_6   = arith.addi %l3_5, %s3_5_2 : tensor<8xi16>
  %s3_6_1 = arith.muli %arg6,  %arg6  : tensor<8xi16>
  %s3_6_2 = arith.muli %s3_6_1, %s3_6_1 : tensor<8xi16>
  %l3_7   = arith.addi %l3_6, %s3_6_2 : tensor<8xi16>
  %s3_7_1 = arith.muli %arg7,  %arg7  : tensor<8xi16>
  %s3_7_2 = arith.muli %s3_7_1, %s3_7_1 : tensor<8xi16>
  %l3_8   = arith.addi %l3_7, %s3_7_2 : tensor<8xi16>
  %s3_8_1 = arith.muli %arg8,  %arg8  : tensor<8xi16>
  %s3_8_2 = arith.muli %s3_8_1, %s3_8_1 : tensor<8xi16>
  %l3_9   = arith.addi %l3_8, %s3_8_2 : tensor<8xi16>
  %s3_9_1 = arith.muli %arg9,  %arg9  : tensor<8xi16>
  %s3_9_2 = arith.muli %s3_9_1, %s3_9_1 : tensor<8xi16>
  %l3_10  = arith.addi %l3_9, %s3_9_2 : tensor<8xi16>
  %s3_10_1 = arith.muli %arg10, %arg10 : tensor<8xi16>
  %s3_10_2 = arith.muli %s3_10_1, %s3_10_1 : tensor<8xi16>
  %l3_11  = arith.addi %l3_10, %s3_10_2 : tensor<8xi16>
  %s3_11_1 = arith.muli %arg11, %arg11 : tensor<8xi16>
  %s3_11_2 = arith.muli %s3_11_1, %s3_11_1 : tensor<8xi16>
  %l3_12  = arith.addi %l3_11, %s3_11_2 : tensor<8xi16>
  %s3_12_1 = arith.muli %arg12, %arg12 : tensor<8xi16>
  %s3_12_2 = arith.muli %s3_12_1, %s3_12_1 : tensor<8xi16>
  %l3_13  = arith.addi %l3_12, %s3_12_2 : tensor<8xi16>
  %s3_13_1 = arith.muli %arg13, %arg13 : tensor<8xi16>
  %s3_13_2 = arith.muli %s3_13_1, %s3_13_1 : tensor<8xi16>
  %l3_14  = arith.addi %l3_13, %s3_13_2 : tensor<8xi16>
  %s3_14_1 = arith.muli %arg14, %arg14 : tensor<8xi16>
  %s3_14_2 = arith.muli %s3_14_1, %s3_14_1 : tensor<8xi16>
  %l3_15  = arith.addi %l3_14, %s3_14_2 : tensor<8xi16>
  %s3_15_1 = arith.muli %arg15, %arg15 : tensor<8xi16>
  %s3_15_2 = arith.muli %s3_15_1, %s3_15_1 : tensor<8xi16>
  %l3_16  = arith.addi %l3_15, %s3_15_2 : tensor<8xi16>
  %s3_16_1 = arith.muli %arg16, %arg16 : tensor<8xi16>
  %s3_16_2 = arith.muli %s3_16_1, %s3_16_1 : tensor<8xi16>
  %l3_17  = arith.addi %l3_16, %s3_16_2 : tensor<8xi16>
  %s3_17_1 = arith.muli %arg17, %arg17 : tensor<8xi16>
  %s3_17_2 = arith.muli %s3_17_1, %s3_17_1 : tensor<8xi16>
  %l3_18  = arith.addi %l3_17, %s3_17_2 : tensor<8xi16>
  %s3_18_1 = arith.muli %arg18, %arg18 : tensor<8xi16>
  %s3_18_2 = arith.muli %s3_18_1, %s3_18_1 : tensor<8xi16>
  %l3_19  = arith.addi %l3_18, %s3_18_2 : tensor<8xi16>
  %s3_19_1 = arith.muli %arg19, %arg19 : tensor<8xi16>
  %s3_19_2 = arith.muli %s3_19_1, %s3_19_1 : tensor<8xi16>
  %l3_20  = arith.addi %l3_19, %s3_19_2 : tensor<8xi16>
  %s3_20_1 = arith.muli %arg20, %arg20 : tensor<8xi16>
  %s3_20_2 = arith.muli %s3_20_1, %s3_20_1 : tensor<8xi16>
  %l3_21  = arith.addi %l3_20, %s3_20_2 : tensor<8xi16>
  %s3_21_1 = arith.muli %arg21, %arg21 : tensor<8xi16>
  %s3_21_2 = arith.muli %s3_21_1, %s3_21_1 : tensor<8xi16>
  %l3_22  = arith.addi %l3_21, %s3_21_2 : tensor<8xi16>
  %s3_22_1 = arith.muli %arg22, %arg22 : tensor<8xi16>
  %s3_22_2 = arith.muli %s3_22_1, %s3_22_1 : tensor<8xi16>
  %l3_23  = arith.addi %l3_22, %s3_22_2 : tensor<8xi16>
  %s3_23_1 = arith.muli %arg23, %arg23 : tensor<8xi16>
  %s3_23_2 = arith.muli %s3_23_1, %s3_23_1 : tensor<8xi16>
  %l3_24  = arith.addi %l3_23, %s3_23_2 : tensor<8xi16>
  %s3_24_1 = arith.muli %arg24, %arg24 : tensor<8xi16>
  %s3_24_2 = arith.muli %s3_24_1, %s3_24_1 : tensor<8xi16>
  %l3_25  = arith.addi %l3_24, %s3_24_2 : tensor<8xi16>
  %s3_25_1 = arith.muli %arg25, %arg25 : tensor<8xi16>
  %s3_25_2 = arith.muli %s3_25_1, %s3_25_1 : tensor<8xi16>
  %l3_26  = arith.addi %l3_25, %s3_25_2 : tensor<8xi16>
  %s3_26_1 = arith.muli %arg26, %arg26 : tensor<8xi16>
  %s3_26_2 = arith.muli %s3_26_1, %s3_26_1 : tensor<8xi16>
  %l3_27  = arith.addi %l3_26, %s3_26_2 : tensor<8xi16>
  %s3_27_1 = arith.muli %arg27, %arg27 : tensor<8xi16>
  %s3_27_2 = arith.muli %s3_27_1, %s3_27_1 : tensor<8xi16>
  %l3_28  = arith.addi %l3_27, %s3_27_2 : tensor<8xi16>
  %s3_28_1 = arith.muli %arg28, %arg28 : tensor<8xi16>
  %s3_28_2 = arith.muli %s3_28_1, %s3_28_1 : tensor<8xi16>
  %l3_29  = arith.addi %l3_28, %s3_28_2 : tensor<8xi16>
  %s3_29_1 = arith.muli %arg29, %arg29 : tensor<8xi16>
  %s3_29_2 = arith.muli %s3_29_1, %s3_29_1 : tensor<8xi16>
  %l3_30  = arith.addi %l3_29, %s3_29_2 : tensor<8xi16>
  %s3_30_1 = arith.muli %arg30, %arg30 : tensor<8xi16>
  %s3_30_2 = arith.muli %s3_30_1, %s3_30_1 : tensor<8xi16>
  %l3_31  = arith.addi %l3_30, %s3_30_2 : tensor<8xi16>
  %s3_31_1 = arith.muli %arg31, %arg31 : tensor<8xi16>
  %s3_31_2 = arith.muli %s3_31_1, %s3_31_1 : tensor<8xi16>
  %l3_32  = arith.addi %l3_31, %s3_31_2 : tensor<8xi16>
  %s3_32_1 = arith.muli %arg32, %arg32 : tensor<8xi16>
  %s3_32_2 = arith.muli %s3_32_1, %s3_32_1 : tensor<8xi16>
  %l3_33  = arith.addi %l3_32, %s3_32_2 : tensor<8xi16>
  %s3_33_1 = arith.muli %arg33, %arg33 : tensor<8xi16>
  %s3_33_2 = arith.muli %s3_33_1, %s3_33_1 : tensor<8xi16>
  %l3_34  = arith.addi %l3_33, %s3_33_2 : tensor<8xi16>
  %s3_34_1 = arith.muli %arg34, %arg34 : tensor<8xi16>
  %s3_34_2 = arith.muli %s3_34_1, %s3_34_1 : tensor<8xi16>
  %l3_35  = arith.addi %l3_34, %s3_34_2 : tensor<8xi16>
  %s3_35_1 = arith.muli %arg35, %arg35 : tensor<8xi16>
  %s3_35_2 = arith.muli %s3_35_1, %s3_35_1 : tensor<8xi16>
  %l3_36  = arith.addi %l3_35, %s3_35_2 : tensor<8xi16>
  %s3_36_1 = arith.muli %arg36, %arg36 : tensor<8xi16>
  %s3_36_2 = arith.muli %s3_36_1, %s3_36_1 : tensor<8xi16>
  %l3_37  = arith.addi %l3_36, %s3_36_2 : tensor<8xi16>
  %s3_37_1 = arith.muli %arg37, %arg37 : tensor<8xi16>
  %s3_37_2 = arith.muli %s3_37_1, %s3_37_1 : tensor<8xi16>
  %l3_38  = arith.addi %l3_37, %s3_37_2 : tensor<8xi16>
  %s3_38_1 = arith.muli %arg38, %arg38 : tensor<8xi16>
  %s3_38_2 = arith.muli %s3_38_1, %s3_38_1 : tensor<8xi16>
  %l3_39  = arith.addi %l3_38, %s3_38_2 : tensor<8xi16>
  %s3_39_1 = arith.muli %arg39, %arg39 : tensor<8xi16>
  %s3_39_2 = arith.muli %s3_39_1, %s3_39_1 : tensor<8xi16>
  %l3_40  = arith.addi %l3_39, %s3_39_2 : tensor<8xi16>
  %s3_40_1 = arith.muli %arg40, %arg40 : tensor<8xi16>
  %s3_40_2 = arith.muli %s3_40_1, %s3_40_1 : tensor<8xi16>
  %l3_41  = arith.addi %l3_40, %s3_40_2 : tensor<8xi16>
  %s3_41_1 = arith.muli %arg41, %arg41 : tensor<8xi16>
  %s3_41_2 = arith.muli %s3_41_1, %s3_41_1 : tensor<8xi16>
  %l3_42  = arith.addi %l3_41, %s3_41_2 : tensor<8xi16>
  %s3_42_1 = arith.muli %arg42, %arg42 : tensor<8xi16>
  %s3_42_2 = arith.muli %s3_42_1, %s3_42_1 : tensor<8xi16>
  %l3_43  = arith.addi %l3_42, %s3_42_2 : tensor<8xi16>
  %s3_43_1 = arith.muli %arg43, %arg43 : tensor<8xi16>
  %s3_43_2 = arith.muli %s3_43_1, %s3_43_1 : tensor<8xi16>
  %l3_44  = arith.addi %l3_43, %s3_43_2 : tensor<8xi16>
  %s3_44_1 = arith.muli %arg44, %arg44 : tensor<8xi16>
  %s3_44_2 = arith.muli %s3_44_1, %s3_44_1 : tensor<8xi16>
  %l3_45  = arith.addi %l3_44, %s3_44_2 : tensor<8xi16>
  %s3_45_1 = arith.muli %arg45, %arg45 : tensor<8xi16>
  %s3_45_2 = arith.muli %s3_45_1, %s3_45_1 : tensor<8xi16>
  %l3_46  = arith.addi %l3_45, %s3_45_2 : tensor<8xi16>
  %s3_46_1 = arith.muli %arg46, %arg46 : tensor<8xi16>
  %s3_46_2 = arith.muli %s3_46_1, %s3_46_1 : tensor<8xi16>
  %l3_47  = arith.addi %l3_46, %s3_46_2 : tensor<8xi16>
  %s3_47_1 = arith.muli %arg47, %arg47 : tensor<8xi16>
  %s3_47_2 = arith.muli %s3_47_1, %s3_47_1 : tensor<8xi16>
  %l3_48  = arith.addi %l3_47, %s3_47_2 : tensor<8xi16>
  %s3_48_1 = arith.muli %arg48, %arg48 : tensor<8xi16>
  %s3_48_2 = arith.muli %s3_48_1, %s3_48_1 : tensor<8xi16>
  %l3_49  = arith.addi %l3_48, %s3_48_2 : tensor<8xi16>
  %s3_49_1 = arith.muli %arg49, %arg49 : tensor<8xi16>
  %s3_49_2 = arith.muli %s3_49_1, %s3_49_1 : tensor<8xi16>
  %l3_50  = arith.addi %l3_49, %s3_49_2 : tensor<8xi16>
  %s3_50_1 = arith.muli %arg50, %arg50 : tensor<8xi16>
  %s3_50_2 = arith.muli %s3_50_1, %s3_50_1 : tensor<8xi16>
  %l3_51  = arith.addi %l3_50, %s3_50_2 : tensor<8xi16>
  %s3_51_1 = arith.muli %arg51, %arg51 : tensor<8xi16>
  %s3_51_2 = arith.muli %s3_51_1, %s3_51_1 : tensor<8xi16>
  %l3_52  = arith.addi %l3_51, %s3_51_2 : tensor<8xi16>
  %s3_52_1 = arith.muli %arg52, %arg52 : tensor<8xi16>
  %s3_52_2 = arith.muli %s3_52_1, %s3_52_1 : tensor<8xi16>
  %l3_53  = arith.addi %l3_52, %s3_52_2 : tensor<8xi16>
  %s3_53_1 = arith.muli %arg53, %arg53 : tensor<8xi16>
  %s3_53_2 = arith.muli %s3_53_1, %s3_53_1 : tensor<8xi16>
  %l3_54  = arith.addi %l3_53, %s3_53_2 : tensor<8xi16>
  %s3_54_1 = arith.muli %arg54, %arg54 : tensor<8xi16>
  %s3_54_2 = arith.muli %s3_54_1, %s3_54_1 : tensor<8xi16>
  %l3_55  = arith.addi %l3_54, %s3_54_2 : tensor<8xi16>
  %s3_55_1 = arith.muli %arg55, %arg55 : tensor<8xi16>
  %s3_55_2 = arith.muli %s3_55_1, %s3_55_1 : tensor<8xi16>
  %l3_56  = arith.addi %l3_55, %s3_55_2 : tensor<8xi16>
  %s3_56_1 = arith.muli %arg56, %arg56 : tensor<8xi16>
  %s3_56_2 = arith.muli %s3_56_1, %s3_56_1 : tensor<8xi16>
  %l3_57  = arith.addi %l3_56, %s3_56_2 : tensor<8xi16>
  %s3_57_1 = arith.muli %arg57, %arg57 : tensor<8xi16>
  %s3_57_2 = arith.muli %s3_57_1, %s3_57_1 : tensor<8xi16>
  %l3_58  = arith.addi %l3_57, %s3_57_2 : tensor<8xi16>
  %s3_58_1 = arith.muli %arg58, %arg58 : tensor<8xi16>
  %s3_58_2 = arith.muli %s3_58_1, %s3_58_1 : tensor<8xi16>
  %l3_59  = arith.addi %l3_58, %s3_58_2 : tensor<8xi16>
  %s3_59_1 = arith.muli %arg59, %arg59 : tensor<8xi16>
  %s3_59_2 = arith.muli %s3_59_1, %s3_59_1 : tensor<8xi16>
  %l3_60  = arith.addi %l3_59, %s3_59_2 : tensor<8xi16>
  %s3_60_1 = arith.muli %arg60, %arg60 : tensor<8xi16>
  %s3_60_2 = arith.muli %s3_60_1, %s3_60_1 : tensor<8xi16>
  %l3_61  = arith.addi %l3_60, %s3_60_2 : tensor<8xi16>
  %s3_61_1 = arith.muli %arg61, %arg61 : tensor<8xi16>
  %s3_61_2 = arith.muli %s3_61_1, %s3_61_1 : tensor<8xi16>
  %l3_62  = arith.addi %l3_61, %s3_61_2 : tensor<8xi16>
  %s3_62_1 = arith.muli %arg62, %arg62 : tensor<8xi16>
  %s3_62_2 = arith.muli %s3_62_1, %s3_62_1 : tensor<8xi16>
  %l3_63  = arith.addi %l3_62, %s3_62_2 : tensor<8xi16>
  %s3_63_1 = arith.muli %arg63, %arg63 : tensor<8xi16>
  %s3_63_2 = arith.muli %s3_63_1, %s3_63_1 : tensor<8xi16>
  %l3_64  = arith.addi %l3_63, %s3_63_2 : tensor<8xi16>
  %m3    = arith.muli %l3_64, %l3_64 : tensor<8xi16>

  //===-------------------------------------------------------------------===
  // Level 4: 16 additions (operands: %arg0 .. %arg15, each squared three times)
  %s4_0_1 = arith.muli %arg0, %arg0 : tensor<8xi16>
  %s4_0_2 = arith.muli %s4_0_1, %s4_0_1 : tensor<8xi16>
  %s4_0_3 = arith.muli %s4_0_2, %s4_0_2 : tensor<8xi16>
  %l4_1   = arith.addi %m3, %s4_0_3 : tensor<8xi16>
  %s4_1_1 = arith.muli %arg1, %arg1 : tensor<8xi16>
  %s4_1_2 = arith.muli %s4_1_1, %s4_1_1 : tensor<8xi16>
  %s4_1_3 = arith.muli %s4_1_2, %s4_1_2 : tensor<8xi16>
  %l4_2   = arith.addi %l4_1, %s4_1_3 : tensor<8xi16>
  %s4_2_1 = arith.muli %arg2, %arg2 : tensor<8xi16>
  %s4_2_2 = arith.muli %s4_2_1, %s4_2_1 : tensor<8xi16>
  %s4_2_3 = arith.muli %s4_2_2, %s4_2_2 : tensor<8xi16>
  %l4_3   = arith.addi %l4_2, %s4_2_3 : tensor<8xi16>
  %s4_3_1 = arith.muli %arg3, %arg3 : tensor<8xi16>
  %s4_3_2 = arith.muli %s4_3_1, %s4_3_1 : tensor<8xi16>
  %s4_3_3 = arith.muli %s4_3_2, %s4_3_2 : tensor<8xi16>
  %l4_4   = arith.addi %l4_3, %s4_3_3 : tensor<8xi16>
  %s4_4_1 = arith.muli %arg4, %arg4 : tensor<8xi16>
  %s4_4_2 = arith.muli %s4_4_1, %s4_4_1 : tensor<8xi16>
  %s4_4_3 = arith.muli %s4_4_2, %s4_4_2 : tensor<8xi16>
  %l4_5   = arith.addi %l4_4, %s4_4_3 : tensor<8xi16>
  %s4_5_1 = arith.muli %arg5, %arg5 : tensor<8xi16>
  %s4_5_2 = arith.muli %s4_5_1, %s4_5_1 : tensor<8xi16>
  %s4_5_3 = arith.muli %s4_5_2, %s4_5_2 : tensor<8xi16>
  %l4_6   = arith.addi %l4_5, %s4_5_3 : tensor<8xi16>
  %s4_6_1 = arith.muli %arg6, %arg6 : tensor<8xi16>
  %s4_6_2 = arith.muli %s4_6_1, %s4_6_1 : tensor<8xi16>
  %s4_6_3 = arith.muli %s4_6_2, %s4_6_2 : tensor<8xi16>
  %l4_7   = arith.addi %l4_6, %s4_6_3 : tensor<8xi16>
  %s4_7_1 = arith.muli %arg7, %arg7 : tensor<8xi16>
  %s4_7_2 = arith.muli %s4_7_1, %s4_7_1 : tensor<8xi16>
  %s4_7_3 = arith.muli %s4_7_2, %s4_7_2 : tensor<8xi16>
  %l4_8   = arith.addi %l4_7, %s4_7_3 : tensor<8xi16>
  %s4_8_1 = arith.muli %arg8, %arg8 : tensor<8xi16>
  %s4_8_2 = arith.muli %s4_8_1, %s4_8_1 : tensor<8xi16>
  %s4_8_3 = arith.muli %s4_8_2, %s4_8_2 : tensor<8xi16>
  %l4_9   = arith.addi %l4_8, %s4_8_3 : tensor<8xi16>
  %s4_9_1 = arith.muli %arg9, %arg9 : tensor<8xi16>
  %s4_9_2 = arith.muli %s4_9_1, %s4_9_1 : tensor<8xi16>
  %s4_9_3 = arith.muli %s4_9_2, %s4_9_2 : tensor<8xi16>
  %l4_10  = arith.addi %l4_9, %s4_9_3 : tensor<8xi16>
  %s4_10_1 = arith.muli %arg10, %arg10 : tensor<8xi16>
  %s4_10_2 = arith.muli %s4_10_1, %s4_10_1 : tensor<8xi16>
  %s4_10_3 = arith.muli %s4_10_2, %s4_10_2 : tensor<8xi16>
  %l4_11  = arith.addi %l4_10, %s4_10_3 : tensor<8xi16>
  %s4_11_1 = arith.muli %arg11, %arg11 : tensor<8xi16>
  %s4_11_2 = arith.muli %s4_11_1, %s4_11_1 : tensor<8xi16>
  %s4_11_3 = arith.muli %s4_11_2, %s4_11_2 : tensor<8xi16>
  %l4_12  = arith.addi %l4_11, %s4_11_3 : tensor<8xi16>
  %s4_12_1 = arith.muli %arg12, %arg12 : tensor<8xi16>
  %s4_12_2 = arith.muli %s4_12_1, %s4_12_1 : tensor<8xi16>
  %s4_12_3 = arith.muli %s4_12_2, %s4_12_2 : tensor<8xi16>
  %l4_13  = arith.addi %l4_12, %s4_12_3 : tensor<8xi16>
  %s4_13_1 = arith.muli %arg13, %arg13 : tensor<8xi16>
  %s4_13_2 = arith.muli %s4_13_1, %s4_13_1 : tensor<8xi16>
  %s4_13_3 = arith.muli %s4_13_2, %s4_13_2 : tensor<8xi16>
  %l4_14  = arith.addi %l4_13, %s4_13_3 : tensor<8xi16>
  %s4_14_1 = arith.muli %arg14, %arg14 : tensor<8xi16>
  %s4_14_2 = arith.muli %s4_14_1, %s4_14_1 : tensor<8xi16>
  %s4_14_3 = arith.muli %s4_14_2, %s4_14_2 : tensor<8xi16>
  %l4_15  = arith.addi %l4_14, %s4_14_3 : tensor<8xi16>
  %s4_15_1 = arith.muli %arg15, %arg15 : tensor<8xi16>
  %s4_15_2 = arith.muli %s4_15_1, %s4_15_1 : tensor<8xi16>
  %s4_15_3 = arith.muli %s4_15_2, %s4_15_2 : tensor<8xi16>
  %l4_16  = arith.addi %l4_15, %s4_15_3 : tensor<8xi16>
  %m4     = arith.muli %l4_16, %l4_16 : tensor<8xi16>

  //===-------------------------------------------------------------------===
  // Level 5: 4 additions (operands: %arg0 .. %arg3, each squared four times)
  %s5_0_1 = arith.muli %arg0, %arg0 : tensor<8xi16>
  %s5_0_2 = arith.muli %s5_0_1, %s5_0_1 : tensor<8xi16>
  %s5_0_3 = arith.muli %s5_0_2, %s5_0_2 : tensor<8xi16>
  %s5_0_4 = arith.muli %s5_0_3, %s5_0_3 : tensor<8xi16>
  %l5_1   = arith.addi %m4, %s5_0_4 : tensor<8xi16>
  %s5_1_1 = arith.muli %arg1, %arg1 : tensor<8xi16>
  %s5_1_2 = arith.muli %s5_1_1, %s5_1_1 : tensor<8xi16>
  %s5_1_3 = arith.muli %s5_1_2, %s5_1_2 : tensor<8xi16>
  %s5_1_4 = arith.muli %s5_1_3, %s5_1_3 : tensor<8xi16>
  %l5_2   = arith.addi %l5_1, %s5_1_4 : tensor<8xi16>
  %s5_2_1 = arith.muli %arg2, %arg2 : tensor<8xi16>
  %s5_2_2 = arith.muli %s5_2_1, %s5_2_1 : tensor<8xi16>
  %s5_2_3 = arith.muli %s5_2_2, %s5_2_2 : tensor<8xi16>
  %s5_2_4 = arith.muli %s5_2_3, %s5_2_3 : tensor<8xi16>
  %l5_3   = arith.addi %l5_2, %s5_2_4 : tensor<8xi16>
  %s5_3_1 = arith.muli %arg3, %arg3 : tensor<8xi16>
  %s5_3_2 = arith.muli %s5_3_1, %s5_3_1 : tensor<8xi16>
  %s5_3_3 = arith.muli %s5_3_2, %s5_3_2 : tensor<8xi16>
  %s5_3_4 = arith.muli %s5_3_3, %s5_3_3 : tensor<8xi16>
  %l5_4   = arith.addi %l5_3, %s5_3_4 : tensor<8xi16>

  return %l5_4 : tensor<8xi16>
}
