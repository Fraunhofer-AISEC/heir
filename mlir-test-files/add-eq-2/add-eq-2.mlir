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
  // Level 1: 0 additions, 1 squaring
  // sq1 = arg0^2
  %sq1 = arith.muli %arg0, %arg0 : tensor<8xi16>

  //-------------------------------------------------------------------------
  // Level 2: 32 squarings + 32 additions
  // sq2 = sq1 + sum_{i=1}^{32} arg_i^2
  %v2_1  = arith.muli %arg1,  %arg1  : tensor<8xi16>
  %v2_2  = arith.muli %arg2,  %arg2  : tensor<8xi16>
  %v2_3  = arith.muli %arg3,  %arg3  : tensor<8xi16>
  %v2_4  = arith.muli %arg4,  %arg4  : tensor<8xi16>
  %v2_5  = arith.muli %arg5,  %arg5  : tensor<8xi16>
  %v2_6  = arith.muli %arg6,  %arg6  : tensor<8xi16>
  %v2_7  = arith.muli %arg7,  %arg7  : tensor<8xi16>
  %v2_8  = arith.muli %arg8,  %arg8  : tensor<8xi16>
  %v2_9  = arith.muli %arg9,  %arg9  : tensor<8xi16>
  %v2_10 = arith.muli %arg10, %arg10 : tensor<8xi16>
  %v2_11 = arith.muli %arg11, %arg11 : tensor<8xi16>
  %v2_12 = arith.muli %arg12, %arg12 : tensor<8xi16>
  %v2_13 = arith.muli %arg13, %arg13 : tensor<8xi16>
  %v2_14 = arith.muli %arg14, %arg14 : tensor<8xi16>
  %v2_15 = arith.muli %arg15, %arg15 : tensor<8xi16>
  %v2_16 = arith.muli %arg16, %arg16 : tensor<8xi16>
  %v2_17 = arith.muli %arg17, %arg17 : tensor<8xi16>
  %v2_18 = arith.muli %arg18, %arg18 : tensor<8xi16>
  %v2_19 = arith.muli %arg19, %arg19 : tensor<8xi16>
  %v2_20 = arith.muli %arg20, %arg20 : tensor<8xi16>
  %v2_21 = arith.muli %arg21, %arg21 : tensor<8xi16>
  %v2_22 = arith.muli %arg22, %arg22 : tensor<8xi16>
  %v2_23 = arith.muli %arg23, %arg23 : tensor<8xi16>
  %v2_24 = arith.muli %arg24, %arg24 : tensor<8xi16>
  %v2_25 = arith.muli %arg25, %arg25 : tensor<8xi16>
  %v2_26 = arith.muli %arg26, %arg26 : tensor<8xi16>
  %v2_27 = arith.muli %arg27, %arg27 : tensor<8xi16>
  %v2_28 = arith.muli %arg28, %arg28 : tensor<8xi16>
  %v2_29 = arith.muli %arg29, %arg29 : tensor<8xi16>
  %v2_30 = arith.muli %arg30, %arg30 : tensor<8xi16>
  %v2_31 = arith.muli %arg31, %arg31 : tensor<8xi16>
  %v2_32 = arith.muli %arg32, %arg32 : tensor<8xi16>

  %l2_0  = arith.addi %sq1,   %v2_1  : tensor<8xi16>
  %l2_1  = arith.addi %l2_0,  %v2_2  : tensor<8xi16>
  %l2_2  = arith.addi %l2_1,  %v2_3  : tensor<8xi16>
  %l2_3  = arith.addi %l2_2,  %v2_4  : tensor<8xi16>
  %l2_4  = arith.addi %l2_3,  %v2_5  : tensor<8xi16>
  %l2_5  = arith.addi %l2_4,  %v2_6  : tensor<8xi16>
  %l2_6  = arith.addi %l2_5,  %v2_7  : tensor<8xi16>
  %l2_7  = arith.addi %l2_6,  %v2_8  : tensor<8xi16>
  %l2_8  = arith.addi %l2_7,  %v2_9  : tensor<8xi16>
  %l2_9  = arith.addi %l2_8,  %v2_10 : tensor<8xi16>
  %l2_10 = arith.addi %l2_9,  %v2_11 : tensor<8xi16>
  %l2_11 = arith.addi %l2_10, %v2_12 : tensor<8xi16>
  %l2_12 = arith.addi %l2_11, %v2_13 : tensor<8xi16>
  %l2_13 = arith.addi %l2_12, %v2_14 : tensor<8xi16>
  %l2_14 = arith.addi %l2_13, %v2_15 : tensor<8xi16>
  %l2_15 = arith.addi %l2_14, %v2_16 : tensor<8xi16>
  %l2_16 = arith.addi %l2_15, %v2_17 : tensor<8xi16>
  %l2_17 = arith.addi %l2_16, %v2_18 : tensor<8xi16>
  %l2_18 = arith.addi %l2_17, %v2_19 : tensor<8xi16>
  %l2_19 = arith.addi %l2_18, %v2_20 : tensor<8xi16>
  %l2_20 = arith.addi %l2_19, %v2_21 : tensor<8xi16>
  %l2_21 = arith.addi %l2_20, %v2_22 : tensor<8xi16>
  %l2_22 = arith.addi %l2_21, %v2_23 : tensor<8xi16>
  %l2_23 = arith.addi %l2_22, %v2_24 : tensor<8xi16>
  %l2_24 = arith.addi %l2_23, %v2_25 : tensor<8xi16>
  %l2_25 = arith.addi %l2_24, %v2_26 : tensor<8xi16>
  %l2_26 = arith.addi %l2_25, %v2_27 : tensor<8xi16>
  %l2_27 = arith.addi %l2_26, %v2_28 : tensor<8xi16>
  %l2_28 = arith.addi %l2_27, %v2_29 : tensor<8xi16>
  %l2_29 = arith.addi %l2_28, %v2_30 : tensor<8xi16>
  %l2_30 = arith.addi %l2_29, %v2_31 : tensor<8xi16>
  %sq2   = arith.addi %l2_30, %v2_32 : tensor<8xi16>

  //-------------------------------------------------------------------------
  // Level 3: 32 squarings (arg_i^4) + 1 muli (sq2 * sq1) + 32 additions
  // sq3 = sq2 * arg0^2 + sum_{i=1}^{32} arg_i^4
  %v4_1  = arith.muli %v2_1,  %v2_1  : tensor<8xi16>
  %v4_2  = arith.muli %v2_2,  %v2_2  : tensor<8xi16>
  %v4_3  = arith.muli %v2_3,  %v2_3  : tensor<8xi16>
  %v4_4  = arith.muli %v2_4,  %v2_4  : tensor<8xi16>
  %v4_5  = arith.muli %v2_5,  %v2_5  : tensor<8xi16>
  %v4_6  = arith.muli %v2_6,  %v2_6  : tensor<8xi16>
  %v4_7  = arith.muli %v2_7,  %v2_7  : tensor<8xi16>
  %v4_8  = arith.muli %v2_8,  %v2_8  : tensor<8xi16>
  %v4_9  = arith.muli %v2_9,  %v2_9  : tensor<8xi16>
  %v4_10 = arith.muli %v2_10, %v2_10 : tensor<8xi16>
  %v4_11 = arith.muli %v2_11, %v2_11 : tensor<8xi16>
  %v4_12 = arith.muli %v2_12, %v2_12 : tensor<8xi16>
  %v4_13 = arith.muli %v2_13, %v2_13 : tensor<8xi16>
  %v4_14 = arith.muli %v2_14, %v2_14 : tensor<8xi16>
  %v4_15 = arith.muli %v2_15, %v2_15 : tensor<8xi16>
  %v4_16 = arith.muli %v2_16, %v2_16 : tensor<8xi16>
  %v4_17 = arith.muli %v2_17, %v2_17 : tensor<8xi16>
  %v4_18 = arith.muli %v2_18, %v2_18 : tensor<8xi16>
  %v4_19 = arith.muli %v2_19, %v2_19 : tensor<8xi16>
  %v4_20 = arith.muli %v2_20, %v2_20 : tensor<8xi16>
  %v4_21 = arith.muli %v2_21, %v2_21 : tensor<8xi16>
  %v4_22 = arith.muli %v2_22, %v2_22 : tensor<8xi16>
  %v4_23 = arith.muli %v2_23, %v2_23 : tensor<8xi16>
  %v4_24 = arith.muli %v2_24, %v2_24 : tensor<8xi16>
  %v4_25 = arith.muli %v2_25, %v2_25 : tensor<8xi16>
  %v4_26 = arith.muli %v2_26, %v2_26 : tensor<8xi16>
  %v4_27 = arith.muli %v2_27, %v2_27 : tensor<8xi16>
  %v4_28 = arith.muli %v2_28, %v2_28 : tensor<8xi16>
  %v4_29 = arith.muli %v2_29, %v2_29 : tensor<8xi16>
  %v4_30 = arith.muli %v2_30, %v2_30 : tensor<8xi16>
  %v4_31 = arith.muli %v2_31, %v2_31 : tensor<8xi16>
  %v4_32 = arith.muli %v2_32, %v2_32 : tensor<8xi16>

  %sq2_x_sq1 = arith.muli %sq2, %sq1 : tensor<8xi16>

  %l3_0  = arith.addi %sq2_x_sq1, %v4_1  : tensor<8xi16>
  %l3_1  = arith.addi %l3_0,  %v4_2  : tensor<8xi16>
  %l3_2  = arith.addi %l3_1,  %v4_3  : tensor<8xi16>
  %l3_3  = arith.addi %l3_2,  %v4_4  : tensor<8xi16>
  %l3_4  = arith.addi %l3_3,  %v4_5  : tensor<8xi16>
  %l3_5  = arith.addi %l3_4,  %v4_6  : tensor<8xi16>
  %l3_6  = arith.addi %l3_5,  %v4_7  : tensor<8xi16>
  %l3_7  = arith.addi %l3_6,  %v4_8  : tensor<8xi16>
  %l3_8  = arith.addi %l3_7,  %v4_9  : tensor<8xi16>
  %l3_9  = arith.addi %l3_8,  %v4_10 : tensor<8xi16>
  %l3_10 = arith.addi %l3_9,  %v4_11 : tensor<8xi16>
  %l3_11 = arith.addi %l3_10, %v4_12 : tensor<8xi16>
  %l3_12 = arith.addi %l3_11, %v4_13 : tensor<8xi16>
  %l3_13 = arith.addi %l3_12, %v4_14 : tensor<8xi16>
  %l3_14 = arith.addi %l3_13, %v4_15 : tensor<8xi16>
  %l3_15 = arith.addi %l3_14, %v4_16 : tensor<8xi16>
  %l3_16 = arith.addi %l3_15, %v4_17 : tensor<8xi16>
  %l3_17 = arith.addi %l3_16, %v4_18 : tensor<8xi16>
  %l3_18 = arith.addi %l3_17, %v4_19 : tensor<8xi16>
  %l3_19 = arith.addi %l3_18, %v4_20 : tensor<8xi16>
  %l3_20 = arith.addi %l3_19, %v4_21 : tensor<8xi16>
  %l3_21 = arith.addi %l3_20, %v4_22 : tensor<8xi16>
  %l3_22 = arith.addi %l3_21, %v4_23 : tensor<8xi16>
  %l3_23 = arith.addi %l3_22, %v4_24 : tensor<8xi16>
  %l3_24 = arith.addi %l3_23, %v4_25 : tensor<8xi16>
  %l3_25 = arith.addi %l3_24, %v4_26 : tensor<8xi16>
  %l3_26 = arith.addi %l3_25, %v4_27 : tensor<8xi16>
  %l3_27 = arith.addi %l3_26, %v4_28 : tensor<8xi16>
  %l3_28 = arith.addi %l3_27, %v4_29 : tensor<8xi16>
  %l3_29 = arith.addi %l3_28, %v4_30 : tensor<8xi16>
  %l3_30 = arith.addi %l3_29, %v4_31 : tensor<8xi16>
  %sq3   = arith.addi %l3_30, %v4_32 : tensor<8xi16>

  return %sq3 : tensor<8xi16>
}