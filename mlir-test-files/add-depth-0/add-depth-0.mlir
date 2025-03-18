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
  // Add all squared tensors together
  %sum1 = arith.addi %arg0, %arg1 : tensor<8xi16>
  %sum2 = arith.addi %sum1, %arg2 : tensor<8xi16>
  %sum3 = arith.addi %sum2, %arg3 : tensor<8xi16>
  %sum4 = arith.addi %sum3, %arg4 : tensor<8xi16>
  %sum5 = arith.addi %sum4, %arg5 : tensor<8xi16>
  %sum6 = arith.addi %sum5, %arg6 : tensor<8xi16>
  %sum7 = arith.addi %sum6, %arg7 : tensor<8xi16>
  %sum8 = arith.addi %sum7, %arg8 : tensor<8xi16>
  %sum9 = arith.addi %sum8, %arg9 : tensor<8xi16>
  %sum10 = arith.addi %sum9, %arg10 : tensor<8xi16>
  %sum11 = arith.addi %sum10, %arg11 : tensor<8xi16>
  %sum12 = arith.addi %sum11, %arg12 : tensor<8xi16>
  %sum13 = arith.addi %sum12, %arg13 : tensor<8xi16>
  %sum14 = arith.addi %sum13, %arg14 : tensor<8xi16>
  %sum15 = arith.addi %sum14, %arg15 : tensor<8xi16>
  %sum16 = arith.addi %sum15, %arg16 : tensor<8xi16>
  %sum17 = arith.addi %sum16, %arg17 : tensor<8xi16>
  %sum18 = arith.addi %sum17, %arg18 : tensor<8xi16>
  %sum19 = arith.addi %sum18, %arg19 : tensor<8xi16>
  %sum20 = arith.addi %sum19, %arg20 : tensor<8xi16>
  %sum21 = arith.addi %sum20, %arg21 : tensor<8xi16>
  %sum22 = arith.addi %sum21, %arg22 : tensor<8xi16>
  %sum23 = arith.addi %sum22, %arg23 : tensor<8xi16>
  %sum24 = arith.addi %sum23, %arg24 : tensor<8xi16>
  %sum25 = arith.addi %sum24, %arg25 : tensor<8xi16>
  %sum26 = arith.addi %sum25, %arg26 : tensor<8xi16>
  %sum27 = arith.addi %sum26, %arg27 : tensor<8xi16>
  %sum28 = arith.addi %sum27, %arg28 : tensor<8xi16>
  %sum29 = arith.addi %sum28, %arg29 : tensor<8xi16>
  %sum30 = arith.addi %sum29, %arg30 : tensor<8xi16>
  %sum31 = arith.addi %sum30, %arg31 : tensor<8xi16>
  %sum32 = arith.addi %sum31, %arg32 : tensor<8xi16>
  %sum33 = arith.addi %sum32, %arg33 : tensor<8xi16>
  %sum34 = arith.addi %sum33, %arg34 : tensor<8xi16>
  %sum35 = arith.addi %sum34, %arg35 : tensor<8xi16>
  %sum36 = arith.addi %sum35, %arg36 : tensor<8xi16>
  %sum37 = arith.addi %sum36, %arg37 : tensor<8xi16>
  %sum38 = arith.addi %sum37, %arg38 : tensor<8xi16>
  %sum39 = arith.addi %sum38, %arg39 : tensor<8xi16>
  %sum40 = arith.addi %sum39, %arg40 : tensor<8xi16>
  %sum41 = arith.addi %sum40, %arg41 : tensor<8xi16>
  %sum42 = arith.addi %sum41, %arg42 : tensor<8xi16>
  %sum43 = arith.addi %sum42, %arg43 : tensor<8xi16>
  %sum44 = arith.addi %sum43, %arg44 : tensor<8xi16>
  %sum45 = arith.addi %sum44, %arg45 : tensor<8xi16>
  %sum46 = arith.addi %sum45, %arg46 : tensor<8xi16>
  %sum47 = arith.addi %sum46, %arg47 : tensor<8xi16>
  %sum48 = arith.addi %sum47, %arg48 : tensor<8xi16>
  %sum49 = arith.addi %sum48, %arg49 : tensor<8xi16>
  %sum50 = arith.addi %sum49, %arg50 : tensor<8xi16>
  %sum51 = arith.addi %sum50, %arg51 : tensor<8xi16>
  %sum52 = arith.addi %sum51, %arg52 : tensor<8xi16>
  %sum53 = arith.addi %sum52, %arg53 : tensor<8xi16>
  %sum54 = arith.addi %sum53, %arg54 : tensor<8xi16>
  %sum55 = arith.addi %sum54, %arg55 : tensor<8xi16>
  %sum56 = arith.addi %sum55, %arg56 : tensor<8xi16>
  %sum57 = arith.addi %sum56, %arg57 : tensor<8xi16>
  %sum58 = arith.addi %sum57, %arg58 : tensor<8xi16>
  %sum59 = arith.addi %sum58, %arg59 : tensor<8xi16>
  %sum60 = arith.addi %sum59, %arg60 : tensor<8xi16>
  %sum61 = arith.addi %sum60, %arg61 : tensor<8xi16>
  %sum62 = arith.addi %sum61, %arg62 : tensor<8xi16>
  %sum63 = arith.addi %sum62, %arg63 : tensor<8xi16>

  return %sum63 : tensor<8xi16>
}