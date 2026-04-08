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
  // --- Level 1: 64 additions ---
  %l1_1  = arith.addi %arg0, %arg1 : tensor<8xi16>
  %l1_2  = arith.addi %l1_1, %arg2 : tensor<8xi16>
  %l1_3  = arith.addi %l1_2, %arg3 : tensor<8xi16>
  %l1_4  = arith.addi %l1_3, %arg4 : tensor<8xi16>
  %l1_5  = arith.addi %l1_4, %arg5 : tensor<8xi16>
  %l1_6  = arith.addi %l1_5, %arg6 : tensor<8xi16>
  %l1_7  = arith.addi %l1_6, %arg7 : tensor<8xi16>
  %l1_8  = arith.addi %l1_7, %arg8 : tensor<8xi16>
  %l1_9  = arith.addi %l1_8, %arg9 : tensor<8xi16>
  %l1_10 = arith.addi %l1_9, %arg10 : tensor<8xi16>
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
  %l1_33 = arith.addi %l1_32, %arg33 : tensor<8xi16>
  %l1_34 = arith.addi %l1_33, %arg34 : tensor<8xi16>
  %l1_35 = arith.addi %l1_34, %arg35 : tensor<8xi16>
  %l1_36 = arith.addi %l1_35, %arg36 : tensor<8xi16>
  %l1_37 = arith.addi %l1_36, %arg37 : tensor<8xi16>
  %l1_38 = arith.addi %l1_37, %arg38 : tensor<8xi16>
  %l1_39 = arith.addi %l1_38, %arg39 : tensor<8xi16>
  %l1_40 = arith.addi %l1_39, %arg40 : tensor<8xi16>
  %l1_41 = arith.addi %l1_40, %arg41 : tensor<8xi16>
  %l1_42 = arith.addi %l1_41, %arg42 : tensor<8xi16>
  %l1_43 = arith.addi %l1_42, %arg43 : tensor<8xi16>
  %l1_44 = arith.addi %l1_43, %arg44 : tensor<8xi16>
  %l1_45 = arith.addi %l1_44, %arg45 : tensor<8xi16>
  %l1_46 = arith.addi %l1_45, %arg46 : tensor<8xi16>
  %l1_47 = arith.addi %l1_46, %arg47 : tensor<8xi16>
  %l1_48 = arith.addi %l1_47, %arg48 : tensor<8xi16>
  %l1_49 = arith.addi %l1_48, %arg49 : tensor<8xi16>
  %l1_50 = arith.addi %l1_49, %arg50 : tensor<8xi16>
  %l1_51 = arith.addi %l1_50, %arg51 : tensor<8xi16>
  %l1_52 = arith.addi %l1_51, %arg52 : tensor<8xi16>
  %l1_53 = arith.addi %l1_52, %arg53 : tensor<8xi16>
  %l1_54 = arith.addi %l1_53, %arg54 : tensor<8xi16>
  %l1_55 = arith.addi %l1_54, %arg55 : tensor<8xi16>
  %l1_56 = arith.addi %l1_55, %arg56 : tensor<8xi16>
  %l1_57 = arith.addi %l1_56, %arg57 : tensor<8xi16>
  %l1_58 = arith.addi %l1_57, %arg58 : tensor<8xi16>
  %l1_59 = arith.addi %l1_58, %arg59 : tensor<8xi16>
  %l1_60 = arith.addi %l1_59, %arg60 : tensor<8xi16>
  %l1_61 = arith.addi %l1_60, %arg61 : tensor<8xi16>
  %l1_62 = arith.addi %l1_61, %arg62 : tensor<8xi16>
  %l1_63 = arith.addi %l1_62, %arg63 : tensor<8xi16>
  %l1_64 = arith.addi %l1_63, %arg0 : tensor<8xi16>
  %l1_65 = arith.addi %l1_64, %arg1 : tensor<8xi16>
  %l1_66 = arith.addi %l1_65, %arg2 : tensor<8xi16>
  %l1_67 = arith.addi %l1_66, %arg3 : tensor<8xi16>
  %l1_68 = arith.addi %l1_67, %arg4 : tensor<8xi16>
  %l1_69 = arith.addi %l1_68, %arg5 : tensor<8xi16>
  %l1_70 = arith.addi %l1_69, %arg6 : tensor<8xi16>
  %l1_71 = arith.addi %l1_70, %arg7 : tensor<8xi16>
  %l1_72 = arith.addi %l1_71, %arg8 : tensor<8xi16>
  %l1_73 = arith.addi %l1_72, %arg9 : tensor<8xi16>
  %l1_74 = arith.addi %l1_73, %arg10 : tensor<8xi16>
  %l1_75 = arith.addi %l1_74, %arg11 : tensor<8xi16>
  %l1_76 = arith.addi %l1_75, %arg12 : tensor<8xi16>
  %l1_77 = arith.addi %l1_76, %arg13 : tensor<8xi16>
  %l1_78 = arith.addi %l1_77, %arg14 : tensor<8xi16>
  %l1_79 = arith.addi %l1_78, %arg15 : tensor<8xi16>
  %l1_80 = arith.addi %l1_79, %l1_79 : tensor<8xi16>
  %l1_81 = arith.addi %l1_80, %arg16 : tensor<8xi16>

  // --- Level 2: Include a multiplication (square the previous result) ---
  %l2 = arith.muli %l1_81, %l1_81 : tensor<8xi16>

  // --- Level 3: Include a multiplication (square the result from Level 2) ---
  %l3 = arith.muli %l2, %l2 : tensor<8xi16>

  // --- Level 4: Include a multiplication (square the result from Level 3) ---
  %l4 = arith.muli %l3, %l3 : tensor<8xi16>

  // --- Level 5: Include a multiplication (square the result from Level 4) ---
  %l5 = arith.muli %l4, %l4 : tensor<8xi16>

  return %l5 : tensor<8xi16>
}