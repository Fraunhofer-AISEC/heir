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

  // Square each input tensor twice (raising to power of 4) before adding
  %sq0_temp = arith.muli %arg0, %arg0 : tensor<8xi16>
  %sq0 = arith.muli %sq0_temp, %sq0_temp : tensor<8xi16>
  %sq1_temp = arith.muli %arg1, %arg1 : tensor<8xi16>
  %sq1 = arith.muli %sq1_temp, %sq1_temp : tensor<8xi16>
  %sq2_temp = arith.muli %arg2, %arg2 : tensor<8xi16>
  %sq2 = arith.muli %sq2_temp, %sq2_temp : tensor<8xi16>
  %sq3_temp = arith.muli %arg3, %arg3 : tensor<8xi16>
  %sq3 = arith.muli %sq3_temp, %sq3_temp : tensor<8xi16>
  %sq4_temp = arith.muli %arg4, %arg4 : tensor<8xi16>
  %sq4 = arith.muli %sq4_temp, %sq4_temp : tensor<8xi16>
  %sq5_temp = arith.muli %arg5, %arg5 : tensor<8xi16>
  %sq5 = arith.muli %sq5_temp, %sq5_temp : tensor<8xi16>
  %sq6_temp = arith.muli %arg6, %arg6 : tensor<8xi16>
  %sq6 = arith.muli %sq6_temp, %sq6_temp : tensor<8xi16>
  %sq7_temp = arith.muli %arg7, %arg7 : tensor<8xi16>
  %sq7 = arith.muli %sq7_temp, %sq7_temp : tensor<8xi16>
  %sq8_temp = arith.muli %arg8, %arg8 : tensor<8xi16>
  %sq8 = arith.muli %sq8_temp, %sq8_temp : tensor<8xi16>
  %sq9_temp = arith.muli %arg9, %arg9 : tensor<8xi16>
  %sq9 = arith.muli %sq9_temp, %sq9_temp : tensor<8xi16>
  %sq10_temp = arith.muli %arg10, %arg10 : tensor<8xi16>
  %sq10 = arith.muli %sq10_temp, %sq10_temp : tensor<8xi16>
  %sq11_temp = arith.muli %arg11, %arg11 : tensor<8xi16>
  %sq11 = arith.muli %sq11_temp, %sq11_temp : tensor<8xi16>
  %sq12_temp = arith.muli %arg12, %arg12 : tensor<8xi16>
  %sq12 = arith.muli %sq12_temp, %sq12_temp : tensor<8xi16>
  %sq13_temp = arith.muli %arg13, %arg13 : tensor<8xi16>
  %sq13 = arith.muli %sq13_temp, %sq13_temp : tensor<8xi16>
  %sq14_temp = arith.muli %arg14, %arg14 : tensor<8xi16>
  %sq14 = arith.muli %sq14_temp, %sq14_temp : tensor<8xi16>
  %sq15_temp = arith.muli %arg15, %arg15 : tensor<8xi16>
  %sq15 = arith.muli %sq15_temp, %sq15_temp : tensor<8xi16>
  %sq16_temp = arith.muli %arg16, %arg16 : tensor<8xi16>
  %sq16 = arith.muli %sq16_temp, %sq16_temp : tensor<8xi16>
  %sq17_temp = arith.muli %arg17, %arg17 : tensor<8xi16>
  %sq17 = arith.muli %sq17_temp, %sq17_temp : tensor<8xi16>
  %sq18_temp = arith.muli %arg18, %arg18 : tensor<8xi16>
  %sq18 = arith.muli %sq18_temp, %sq18_temp : tensor<8xi16>
  %sq19_temp = arith.muli %arg19, %arg19 : tensor<8xi16>
  %sq19 = arith.muli %sq19_temp, %sq19_temp : tensor<8xi16>
  %sq20_temp = arith.muli %arg20, %arg20 : tensor<8xi16>
  %sq20 = arith.muli %sq20_temp, %sq20_temp : tensor<8xi16>
  %sq21_temp = arith.muli %arg21, %arg21 : tensor<8xi16>
  %sq21 = arith.muli %sq21_temp, %sq21_temp : tensor<8xi16>
  %sq22_temp = arith.muli %arg22, %arg22 : tensor<8xi16>
  %sq22 = arith.muli %sq22_temp, %sq22_temp : tensor<8xi16>
  %sq23_temp = arith.muli %arg23, %arg23 : tensor<8xi16>
  %sq23 = arith.muli %sq23_temp, %sq23_temp : tensor<8xi16>
  %sq24_temp = arith.muli %arg24, %arg24 : tensor<8xi16>
  %sq24 = arith.muli %sq24_temp, %sq24_temp : tensor<8xi16>
  %sq25_temp = arith.muli %arg25, %arg25 : tensor<8xi16>
  %sq25 = arith.muli %sq25_temp, %sq25_temp : tensor<8xi16>
  %sq26_temp = arith.muli %arg26, %arg26 : tensor<8xi16>
  %sq26 = arith.muli %sq26_temp, %sq26_temp : tensor<8xi16>
  %sq27_temp = arith.muli %arg27, %arg27 : tensor<8xi16>
  %sq27 = arith.muli %sq27_temp, %sq27_temp : tensor<8xi16>
  %sq28_temp = arith.muli %arg28, %arg28 : tensor<8xi16>
  %sq28 = arith.muli %sq28_temp, %sq28_temp : tensor<8xi16>
  %sq29_temp = arith.muli %arg29, %arg29 : tensor<8xi16>
  %sq29 = arith.muli %sq29_temp, %sq29_temp : tensor<8xi16>
  %sq30_temp = arith.muli %arg30, %arg30 : tensor<8xi16>
  %sq30 = arith.muli %sq30_temp, %sq30_temp : tensor<8xi16>
  %sq31_temp = arith.muli %arg31, %arg31 : tensor<8xi16>
  %sq31 = arith.muli %sq31_temp, %sq31_temp : tensor<8xi16>
  %sq32_temp = arith.muli %arg32, %arg32 : tensor<8xi16>
  %sq32 = arith.muli %sq32_temp, %sq32_temp : tensor<8xi16>
  %sq33_temp = arith.muli %arg33, %arg33 : tensor<8xi16>
  %sq33 = arith.muli %sq33_temp, %sq33_temp : tensor<8xi16>
  %sq34_temp = arith.muli %arg34, %arg34 : tensor<8xi16>
  %sq34 = arith.muli %sq34_temp, %sq34_temp : tensor<8xi16>
  %sq35_temp = arith.muli %arg35, %arg35 : tensor<8xi16>
  %sq35 = arith.muli %sq35_temp, %sq35_temp : tensor<8xi16>
  %sq36_temp = arith.muli %arg36, %arg36 : tensor<8xi16>
  %sq36 = arith.muli %sq36_temp, %sq36_temp : tensor<8xi16>
  %sq37_temp = arith.muli %arg37, %arg37 : tensor<8xi16>
  %sq37 = arith.muli %sq37_temp, %sq37_temp : tensor<8xi16>
  %sq38_temp = arith.muli %arg38, %arg38 : tensor<8xi16>
  %sq38 = arith.muli %sq38_temp, %sq38_temp : tensor<8xi16>
  %sq39_temp = arith.muli %arg39, %arg39 : tensor<8xi16>
  %sq39 = arith.muli %sq39_temp, %sq39_temp : tensor<8xi16>
  %sq40_temp = arith.muli %arg40, %arg40 : tensor<8xi16>
  %sq40 = arith.muli %sq40_temp, %sq40_temp : tensor<8xi16>
  %sq41_temp = arith.muli %arg41, %arg41 : tensor<8xi16>
  %sq41 = arith.muli %sq41_temp, %sq41_temp : tensor<8xi16>
  %sq42_temp = arith.muli %arg42, %arg42 : tensor<8xi16>
  %sq42 = arith.muli %sq42_temp, %sq42_temp : tensor<8xi16>
  %sq43_temp = arith.muli %arg43, %arg43 : tensor<8xi16>
  %sq43 = arith.muli %sq43_temp, %sq43_temp : tensor<8xi16>
  %sq44_temp = arith.muli %arg44, %arg44 : tensor<8xi16>
  %sq44 = arith.muli %sq44_temp, %sq44_temp : tensor<8xi16>
  %sq45_temp = arith.muli %arg45, %arg45 : tensor<8xi16>
  %sq45 = arith.muli %sq45_temp, %sq45_temp : tensor<8xi16>
  %sq46_temp = arith.muli %arg46, %arg46 : tensor<8xi16>
  %sq46 = arith.muli %sq46_temp, %sq46_temp : tensor<8xi16>
  %sq47_temp = arith.muli %arg47, %arg47 : tensor<8xi16>
  %sq47 = arith.muli %sq47_temp, %sq47_temp : tensor<8xi16>
  %sq48_temp = arith.muli %arg48, %arg48 : tensor<8xi16>
  %sq48 = arith.muli %sq48_temp, %sq48_temp : tensor<8xi16>
  %sq49_temp = arith.muli %arg49, %arg49 : tensor<8xi16>
  %sq49 = arith.muli %sq49_temp, %sq49_temp : tensor<8xi16>
  %sq50_temp = arith.muli %arg50, %arg50 : tensor<8xi16>
  %sq50 = arith.muli %sq50_temp, %sq50_temp : tensor<8xi16>
  %sq51_temp = arith.muli %arg51, %arg51 : tensor<8xi16>
  %sq51 = arith.muli %sq51_temp, %sq51_temp : tensor<8xi16>
  %sq52_temp = arith.muli %arg52, %arg52 : tensor<8xi16>
  %sq52 = arith.muli %sq52_temp, %sq52_temp : tensor<8xi16>
  %sq53_temp = arith.muli %arg53, %arg53 : tensor<8xi16>
  %sq53 = arith.muli %sq53_temp, %sq53_temp : tensor<8xi16>
  %sq54_temp = arith.muli %arg54, %arg54 : tensor<8xi16>
  %sq54 = arith.muli %sq54_temp, %sq54_temp : tensor<8xi16>
  %sq55_temp = arith.muli %arg55, %arg55 : tensor<8xi16>
  %sq55 = arith.muli %sq55_temp, %sq55_temp : tensor<8xi16>
  %sq56_temp = arith.muli %arg56, %arg56 : tensor<8xi16>
  %sq56 = arith.muli %sq56_temp, %sq56_temp : tensor<8xi16>
  %sq57_temp = arith.muli %arg57, %arg57 : tensor<8xi16>
  %sq57 = arith.muli %sq57_temp, %sq57_temp : tensor<8xi16>
  %sq58_temp = arith.muli %arg58, %arg58 : tensor<8xi16>
  %sq58 = arith.muli %sq58_temp, %sq58_temp : tensor<8xi16>
  %sq59_temp = arith.muli %arg59, %arg59 : tensor<8xi16>
  %sq59 = arith.muli %sq59_temp, %sq59_temp : tensor<8xi16>
  %sq60_temp = arith.muli %arg60, %arg60 : tensor<8xi16>
  %sq60 = arith.muli %sq60_temp, %sq60_temp : tensor<8xi16>
  %sq61_temp = arith.muli %arg61, %arg61 : tensor<8xi16>
  %sq61 = arith.muli %sq61_temp, %sq61_temp : tensor<8xi16>
  %sq62_temp = arith.muli %arg62, %arg62 : tensor<8xi16>
  %sq62 = arith.muli %sq62_temp, %sq62_temp : tensor<8xi16>
  %sq63_temp = arith.muli %arg63, %arg63 : tensor<8xi16>
  %sq63 = arith.muli %sq63_temp, %sq63_temp : tensor<8xi16>

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

  %mult1 = arith.muli %sum63, %sum63 : tensor<8xi16>
  %mult2 = arith.muli %mult1, %mult1 : tensor<8xi16>

  return %mult2 : tensor<8xi16>
}