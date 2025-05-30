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

// Square each input tensor before adding
  %sq0 = arith.muli %arg0, %arg0 : tensor<8xi16>
  %sq1 = arith.muli %arg1, %arg1 : tensor<8xi16>
  %sq2 = arith.muli %arg2, %arg2 : tensor<8xi16>
  %sq3 = arith.muli %arg3, %arg3 : tensor<8xi16>
  %sq4 = arith.muli %arg4, %arg4 : tensor<8xi16>
  %sq5 = arith.muli %arg5, %arg5 : tensor<8xi16>
  %sq6 = arith.muli %arg6, %arg6 : tensor<8xi16>
  %sq7 = arith.muli %arg7, %arg7 : tensor<8xi16>
  %sq8 = arith.muli %arg8, %arg8 : tensor<8xi16>
  %sq9 = arith.muli %arg9, %arg9 : tensor<8xi16>
  %sq10 = arith.muli %arg10, %arg10 : tensor<8xi16>
  %sq11 = arith.muli %arg11, %arg11 : tensor<8xi16>
  %sq12 = arith.muli %arg12, %arg12 : tensor<8xi16>
  %sq13 = arith.muli %arg13, %arg13 : tensor<8xi16>
  %sq14 = arith.muli %arg14, %arg14 : tensor<8xi16>
  %sq15 = arith.muli %arg15, %arg15 : tensor<8xi16>
  %sq16 = arith.muli %arg16, %arg16 : tensor<8xi16>
  %sq17 = arith.muli %arg17, %arg17 : tensor<8xi16>
  %sq18 = arith.muli %arg18, %arg18 : tensor<8xi16>
  %sq19 = arith.muli %arg19, %arg19 : tensor<8xi16>
  %sq20 = arith.muli %arg20, %arg20 : tensor<8xi16>
  %sq21 = arith.muli %arg21, %arg21 : tensor<8xi16>
  %sq22 = arith.muli %arg22, %arg22 : tensor<8xi16>
  %sq23 = arith.muli %arg23, %arg23 : tensor<8xi16>
  %sq24 = arith.muli %arg24, %arg24 : tensor<8xi16>
  %sq25 = arith.muli %arg25, %arg25 : tensor<8xi16>
  %sq26 = arith.muli %arg26, %arg26 : tensor<8xi16>
  %sq27 = arith.muli %arg27, %arg27 : tensor<8xi16>
  %sq28 = arith.muli %arg28, %arg28 : tensor<8xi16>
  %sq29 = arith.muli %arg29, %arg29 : tensor<8xi16>
  %sq30 = arith.muli %arg30, %arg30 : tensor<8xi16>
  %sq31 = arith.muli %arg31, %arg31 : tensor<8xi16>
  %sq32 = arith.muli %arg32, %arg32 : tensor<8xi16>
  %sq33 = arith.muli %arg33, %arg33 : tensor<8xi16>
  %sq34 = arith.muli %arg34, %arg34 : tensor<8xi16>
  %sq35 = arith.muli %arg35, %arg35 : tensor<8xi16>
  %sq36 = arith.muli %arg36, %arg36 : tensor<8xi16>
  %sq37 = arith.muli %arg37, %arg37 : tensor<8xi16>
  %sq38 = arith.muli %arg38, %arg38 : tensor<8xi16>
  %sq39 = arith.muli %arg39, %arg39 : tensor<8xi16>
  %sq40 = arith.muli %arg40, %arg40 : tensor<8xi16>
  %sq41 = arith.muli %arg41, %arg41 : tensor<8xi16>
  %sq42 = arith.muli %arg42, %arg42 : tensor<8xi16>
  %sq43 = arith.muli %arg43, %arg43 : tensor<8xi16>
  %sq44 = arith.muli %arg44, %arg44 : tensor<8xi16>
  %sq45 = arith.muli %arg45, %arg45 : tensor<8xi16>
  %sq46 = arith.muli %arg46, %arg46 : tensor<8xi16>
  %sq47 = arith.muli %arg47, %arg47 : tensor<8xi16>
  %sq48 = arith.muli %arg48, %arg48 : tensor<8xi16>
  %sq49 = arith.muli %arg49, %arg49 : tensor<8xi16>
  %sq50 = arith.muli %arg50, %arg50 : tensor<8xi16>
  %sq51 = arith.muli %arg51, %arg51 : tensor<8xi16>
  %sq52 = arith.muli %arg52, %arg52 : tensor<8xi16>
  %sq53 = arith.muli %arg53, %arg53 : tensor<8xi16>
  %sq54 = arith.muli %arg54, %arg54 : tensor<8xi16>
  %sq55 = arith.muli %arg55, %arg55 : tensor<8xi16>
  %sq56 = arith.muli %arg56, %arg56 : tensor<8xi16>
  %sq57 = arith.muli %arg57, %arg57 : tensor<8xi16>
  %sq58 = arith.muli %arg58, %arg58 : tensor<8xi16>
  %sq59 = arith.muli %arg59, %arg59 : tensor<8xi16>
  %sq60 = arith.muli %arg60, %arg60 : tensor<8xi16>
  %sq61 = arith.muli %arg61, %arg61 : tensor<8xi16>
  %sq62 = arith.muli %arg62, %arg62 : tensor<8xi16>
  %sq63 = arith.muli %arg63, %arg63 : tensor<8xi16>

  // Add all squared tensors together
  %sum1 = arith.addi %sq0, %sq1 : tensor<8xi16>
  %sum2 = arith.addi %sum1, %sq2 : tensor<8xi16>
  %sum3 = arith.addi %sum2, %sq3 : tensor<8xi16>
  %sum4 = arith.addi %sum3, %sq4 : tensor<8xi16>
  %sum5 = arith.addi %sum4, %sq5 : tensor<8xi16>
  %sum6 = arith.addi %sum5, %sq6 : tensor<8xi16>
  %sum7 = arith.addi %sum6, %sq7 : tensor<8xi16>
  %sum8 = arith.addi %sum7, %sq8 : tensor<8xi16>
  %sum9 = arith.addi %sum8, %sq9 : tensor<8xi16>
  %sum10 = arith.addi %sum9, %sq10 : tensor<8xi16>
  %sum11 = arith.addi %sum10, %sq11 : tensor<8xi16>
  %sum12 = arith.addi %sum11, %sq12 : tensor<8xi16>
  %sum13 = arith.addi %sum12, %sq13 : tensor<8xi16>
  %sum14 = arith.addi %sum13, %sq14 : tensor<8xi16>
  %sum15 = arith.addi %sum14, %sq15 : tensor<8xi16>
  %sum16 = arith.addi %sum15, %sq16 : tensor<8xi16>
  %sum17 = arith.addi %sum16, %sq17 : tensor<8xi16>
  %sum18 = arith.addi %sum17, %sq18 : tensor<8xi16>
  %sum19 = arith.addi %sum18, %sq19 : tensor<8xi16>
  %sum20 = arith.addi %sum19, %sq20 : tensor<8xi16>
  %sum21 = arith.addi %sum20, %sq21 : tensor<8xi16>
  %sum22 = arith.addi %sum21, %sq22 : tensor<8xi16>
  %sum23 = arith.addi %sum22, %sq23 : tensor<8xi16>
  %sum24 = arith.addi %sum23, %sq24 : tensor<8xi16>
  %sum25 = arith.addi %sum24, %sq25 : tensor<8xi16>
  %sum26 = arith.addi %sum25, %sq26 : tensor<8xi16>
  %sum27 = arith.addi %sum26, %sq27 : tensor<8xi16>
  %sum28 = arith.addi %sum27, %sq28 : tensor<8xi16>
  %sum29 = arith.addi %sum28, %sq29 : tensor<8xi16>
  %sum30 = arith.addi %sum29, %sq30 : tensor<8xi16>
  %sum31 = arith.addi %sum30, %sq31 : tensor<8xi16>
  %sum32 = arith.addi %sum31, %sq32 : tensor<8xi16>
  %sum33 = arith.addi %sum32, %sq33 : tensor<8xi16>
  %sum34 = arith.addi %sum33, %sq34 : tensor<8xi16>
  %sum35 = arith.addi %sum34, %sq35 : tensor<8xi16>
  %sum36 = arith.addi %sum35, %sq36 : tensor<8xi16>
  %sum37 = arith.addi %sum36, %sq37 : tensor<8xi16>
  %sum38 = arith.addi %sum37, %sq38 : tensor<8xi16>
  %sum39 = arith.addi %sum38, %sq39 : tensor<8xi16>
  %sum40 = arith.addi %sum39, %sq40 : tensor<8xi16>
  %sum41 = arith.addi %sum40, %sq41 : tensor<8xi16>
  %sum42 = arith.addi %sum41, %sq42 : tensor<8xi16>
  %sum43 = arith.addi %sum42, %sq43 : tensor<8xi16>
  %sum44 = arith.addi %sum43, %sq44 : tensor<8xi16>
  %sum45 = arith.addi %sum44, %sq45 : tensor<8xi16>
  %sum46 = arith.addi %sum45, %sq46 : tensor<8xi16>
  %sum47 = arith.addi %sum46, %sq47 : tensor<8xi16>
  %sum48 = arith.addi %sum47, %sq48 : tensor<8xi16>
  %sum49 = arith.addi %sum48, %sq49 : tensor<8xi16>
  %sum50 = arith.addi %sum49, %sq50 : tensor<8xi16>
  %sum51 = arith.addi %sum50, %sq51 : tensor<8xi16>
  %sum52 = arith.addi %sum51, %sq52 : tensor<8xi16>
  %sum53 = arith.addi %sum52, %sq53 : tensor<8xi16>
  %sum54 = arith.addi %sum53, %sq54 : tensor<8xi16>
  %sum55 = arith.addi %sum54, %sq55 : tensor<8xi16>
  %sum56 = arith.addi %sum55, %sq56 : tensor<8xi16>
  %sum57 = arith.addi %sum56, %sq57 : tensor<8xi16>
  %sum58 = arith.addi %sum57, %sq58 : tensor<8xi16>
  %sum59 = arith.addi %sum58, %sq59 : tensor<8xi16>
  %sum60 = arith.addi %sum59, %sq60 : tensor<8xi16>
  %sum61 = arith.addi %sum60, %sq61 : tensor<8xi16>
  %sum62 = arith.addi %sum61, %sq62 : tensor<8xi16>
  %sum63 = arith.addi %sum62, %sq63 : tensor<8xi16>

  return %sum63 : tensor<8xi16>
}