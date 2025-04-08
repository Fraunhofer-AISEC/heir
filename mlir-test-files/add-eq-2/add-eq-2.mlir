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
  //-------------------------------------------------------------------------
  // Level 1: 32 additions (0 squarings)
  // Using operands: %arg0 ... %arg32 (33 operands for 32 additions)
  %l1_1  = arith.addi %arg0, %arg1 : tensor<8xi16>
  %l1_2  = arith.addi %l1_1,  %arg2 : tensor<8xi16>
  %l1_3  = arith.addi %l1_2,  %arg3 : tensor<8xi16>
  %l1_4  = arith.addi %l1_3,  %arg4 : tensor<8xi16>
  %l1_5  = arith.addi %l1_4,  %arg5 : tensor<8xi16>
  %l1_6  = arith.addi %l1_5,  %arg6 : tensor<8xi16>
  %l1_7  = arith.addi %l1_6,  %arg7 : tensor<8xi16>
  %l1_8  = arith.addi %l1_7,  %arg8 : tensor<8xi16>
  %l1_9  = arith.addi %l1_8,  %arg9 : tensor<8xi16>
  %l1_10 = arith.addi %l1_9,  %arg10 : tensor<8xi16>
  %l1_11 = arith.addi %l1_10, %arg11 : tensor<8xi16>
  %l1_12 = arith.addi %l1_11, %arg12 : tensor<8xi16>
  %l1_13 = arith.addi %l1_12, %arg13 : tensor<8xi16>
  %l1_14 = arith.addi %l1_13, %arg14 : tensor<8xi16>
  %l1_15 = arith.addi %l1_14, %arg15 : tensor<8xi16>
  %l1_16 = arith.addi %l1_15, %arg16 : tensor<8xi16>
  %l1_17 = arith.addi %l1_16, %arg17 : tensor<8xi16>
  %l1_18 = arith.addi %l1_17, %arg18 : tensor<8xi16>
  %l1_19 = arith.addi %l1_18, %arg19 : tensor<8xi16>
  %l1_20 = arith.addi %l1_19, %arg20 : tensor<8xi16>
  %l1_21 = arith.addi %l1_20, %arg21 : tensor<8xi16>
  %l1_22 = arith.addi %l1_21, %arg22 : tensor<8xi16>
  %l1_23 = arith.addi %l1_22, %arg23 : tensor<8xi16>
  %l1_24 = arith.addi %l1_23, %arg24 : tensor<8xi16>
  %l1_25 = arith.addi %l1_24, %arg25 : tensor<8xi16>
  %l1_26 = arith.addi %l1_25, %arg26 : tensor<8xi16>
  %l1_27 = arith.addi %l1_26, %arg27 : tensor<8xi16>
  %l1_28 = arith.addi %l1_27, %arg28 : tensor<8xi16>
  %l1_29 = arith.addi %l1_28, %arg29 : tensor<8xi16>
  %l1_30 = arith.addi %l1_29, %arg30 : tensor<8xi16>
  %l1_31 = arith.addi %l1_30, %arg31 : tensor<8xi16>
  %l1_32 = arith.addi %l1_31, %arg32 : tensor<8xi16>
  %m1    = arith.muli %l1_32, %l1_32 : tensor<8xi16>

  //-------------------------------------------------------------------------
  // Level 2: 32 additions (1 squaring)
  // Using operands: %arg0 .. %arg31
  %s2_0  = arith.muli %arg0,  %arg0  : tensor<8xi16>
  %l2_1  = arith.addi %m1,     %s2_0  : tensor<8xi16>
  %s2_1  = arith.muli %arg1,  %arg1  : tensor<8xi16>
  %l2_2  = arith.addi %l2_1,    %s2_1  : tensor<8xi16>
  %s2_2  = arith.muli %arg2,  %arg2  : tensor<8xi16>
  %l2_3  = arith.addi %l2_2,    %s2_2  : tensor<8xi16>
  %s2_3  = arith.muli %arg3,  %arg3  : tensor<8xi16>
  %l2_4  = arith.addi %l2_3,    %s2_3  : tensor<8xi16>
  %s2_4  = arith.muli %arg4,  %arg4  : tensor<8xi16>
  %l2_5  = arith.addi %l2_4,    %s2_4  : tensor<8xi16>
  %s2_5  = arith.muli %arg5,  %arg5  : tensor<8xi16>
  %l2_6  = arith.addi %l2_5,    %s2_5  : tensor<8xi16>
  %s2_6  = arith.muli %arg6,  %arg6  : tensor<8xi16>
  %l2_7  = arith.addi %l2_6,    %s2_6  : tensor<8xi16>
  %s2_7  = arith.muli %arg7,  %arg7  : tensor<8xi16>
  %l2_8  = arith.addi %l2_7,    %s2_7  : tensor<8xi16>
  %s2_8  = arith.muli %arg8,  %arg8  : tensor<8xi16>
  %l2_9  = arith.addi %l2_8,    %s2_8  : tensor<8xi16>
  %s2_9  = arith.muli %arg9,  %arg9  : tensor<8xi16>
  %l2_10 = arith.addi %l2_9,    %s2_9  : tensor<8xi16>
  %s2_10 = arith.muli %arg10, %arg10 : tensor<8xi16>
  %l2_11 = arith.addi %l2_10,   %s2_10 : tensor<8xi16>
  %s2_11 = arith.muli %arg11, %arg11 : tensor<8xi16>
  %l2_12 = arith.addi %l2_11,   %s2_11 : tensor<8xi16>
  %s2_12 = arith.muli %arg12, %arg12 : tensor<8xi16>
  %l2_13 = arith.addi %l2_12,   %s2_12 : tensor<8xi16>
  %s2_13 = arith.muli %arg13, %arg13 : tensor<8xi16>
  %l2_14 = arith.addi %l2_13,   %s2_13 : tensor<8xi16>
  %s2_14 = arith.muli %arg14, %arg14 : tensor<8xi16>
  %l2_15 = arith.addi %l2_14,   %s2_14 : tensor<8xi16>
  %s2_15 = arith.muli %arg15, %arg15 : tensor<8xi16>
  %l2_16 = arith.addi %l2_15,   %s2_15 : tensor<8xi16>
  %s2_16 = arith.muli %arg16, %arg16 : tensor<8xi16>
  %l2_17 = arith.addi %l2_16,   %s2_16 : tensor<8xi16>
  %s2_17 = arith.muli %arg17, %arg17 : tensor<8xi16>
  %l2_18 = arith.addi %l2_17,   %s2_17 : tensor<8xi16>
  %s2_18 = arith.muli %arg18, %arg18 : tensor<8xi16>
  %l2_19 = arith.addi %l2_18,   %s2_18 : tensor<8xi16>
  %s2_19 = arith.muli %arg19, %arg19 : tensor<8xi16>
  %l2_20 = arith.addi %l2_19,   %s2_19 : tensor<8xi16>
  %s2_20 = arith.muli %arg20, %arg20 : tensor<8xi16>
  %l2_21 = arith.addi %l2_20,   %s2_20 : tensor<8xi16>
  %s2_21 = arith.muli %arg21, %arg21 : tensor<8xi16>
  %l2_22 = arith.addi %l2_21,   %s2_21 : tensor<8xi16>
  %s2_22 = arith.muli %arg22, %arg22 : tensor<8xi16>
  %l2_23 = arith.addi %l2_22,   %s2_22 : tensor<8xi16>
  %s2_23 = arith.muli %arg23, %arg23 : tensor<8xi16>
  %l2_24 = arith.addi %l2_23,   %s2_23 : tensor<8xi16>
  %s2_24 = arith.muli %arg24, %arg24 : tensor<8xi16>
  %l2_25 = arith.addi %l2_24,   %s2_24 : tensor<8xi16>
  %s2_25 = arith.muli %arg25, %arg25 : tensor<8xi16>
  %l2_26 = arith.addi %l2_25,   %s2_25 : tensor<8xi16>
  %s2_26 = arith.muli %arg26, %arg26 : tensor<8xi16>
  %l2_27 = arith.addi %l2_26,   %s2_26 : tensor<8xi16>
  %s2_27 = arith.muli %arg27, %arg27 : tensor<8xi16>
  %l2_28 = arith.addi %l2_27,   %s2_27 : tensor<8xi16>
  %s2_28 = arith.muli %arg28, %arg28 : tensor<8xi16>
  %l2_29 = arith.addi %l2_28,   %s2_28 : tensor<8xi16>
  %s2_29 = arith.muli %arg29, %arg29 : tensor<8xi16>
  %l2_30 = arith.addi %l2_29,   %s2_29 : tensor<8xi16>
  %s2_30 = arith.muli %arg30, %arg30 : tensor<8xi16>
  %l2_31 = arith.addi %l2_30,   %s2_30 : tensor<8xi16>
  %s2_31 = arith.muli %arg31, %arg31 : tensor<8xi16>
  %l2_32 = arith.addi %l2_31,   %s2_31 : tensor<8xi16>
  %m2    = arith.muli %l2_32,   %l2_32 : tensor<8xi16>

  return %m2 : tensor<8xi16>
}
