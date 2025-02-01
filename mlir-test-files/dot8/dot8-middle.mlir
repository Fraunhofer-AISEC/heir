!Z1005037682689_i64_ = !mod_arith.int<1005037682689 : i64>
!Z1032955396097_i64_ = !mod_arith.int<1032955396097 : i64>
!Z1095233372161_i64_ = !mod_arith.int<1095233372161 : i64>
!Z4295294977_i64_ = !mod_arith.int<4295294977 : i64>
#full_crt_packing_encoding = #lwe.full_crt_packing_encoding<scaling_factor = 0>
#key = #lwe.key<>
#modulus_chain_L5_C0_ = #lwe.modulus_chain<elements = <1095233372161 : i64, 1032955396097 : i64, 1005037682689 : i64, 998595133441 : i64, 972824936449 : i64, 959939837953 : i64>, current = 0>
#modulus_chain_L5_C1_ = #lwe.modulus_chain<elements = <1095233372161 : i64, 1032955396097 : i64, 1005037682689 : i64, 998595133441 : i64, 972824936449 : i64, 959939837953 : i64>, current = 1>
#modulus_chain_L5_C2_ = #lwe.modulus_chain<elements = <1095233372161 : i64, 1032955396097 : i64, 1005037682689 : i64, 998595133441 : i64, 972824936449 : i64, 959939837953 : i64>, current = 2>
!rns_L0_ = !rns.rns<!Z1095233372161_i64_>
!rns_L1_ = !rns.rns<!Z1095233372161_i64_, !Z1032955396097_i64_>
!rns_L2_ = !rns.rns<!Z1095233372161_i64_, !Z1032955396097_i64_, !Z1005037682689_i64_>
#ring_Z4295294977_i64_1_x8_ = #polynomial.ring<coefficientType = !Z4295294977_i64_, polynomialModulus = <1 + x**8>>
#plaintext_space = #lwe.plaintext_space<ring = #ring_Z4295294977_i64_1_x8_, encoding = #full_crt_packing_encoding>
#ring_rns_L0_1_x8_ = #polynomial.ring<coefficientType = !rns_L0_, polynomialModulus = <1 + x**8>>
#ring_rns_L1_1_x8_ = #polynomial.ring<coefficientType = !rns_L1_, polynomialModulus = <1 + x**8>>
#ring_rns_L2_1_x8_ = #polynomial.ring<coefficientType = !rns_L2_, polynomialModulus = <1 + x**8>>
!pkey_L2_ = !lwe.new_lwe_public_key<key = #key, ring = #ring_rns_L2_1_x8_>
!pt = !lwe.new_lwe_plaintext<application_data = <message_type = tensor<8xi16>>, plaintext_space = #plaintext_space>
!pt1 = !lwe.new_lwe_plaintext<application_data = <message_type = i16>, plaintext_space = #plaintext_space>
!skey_L0_ = !lwe.new_lwe_secret_key<key = #key, ring = #ring_rns_L0_1_x8_>
#ciphertext_space_L0_ = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x8_, encryption_type = lsb>
#ciphertext_space_L1_ = #lwe.ciphertext_space<ring = #ring_rns_L1_1_x8_, encryption_type = lsb>
#ciphertext_space_L2_ = #lwe.ciphertext_space<ring = #ring_rns_L2_1_x8_, encryption_type = lsb>
#ciphertext_space_L2_D3_ = #lwe.ciphertext_space<ring = #ring_rns_L2_1_x8_, encryption_type = lsb, size = 3>
!ct_L0_ = !lwe.new_lwe_ciphertext<application_data = <message_type = i16>, plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L0_, key = #key, modulus_chain = #modulus_chain_L5_C0_>
!ct_L1_ = !lwe.new_lwe_ciphertext<application_data = <message_type = tensor<8xi16>>, plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L1_, key = #key, modulus_chain = #modulus_chain_L5_C1_>
!ct_L1_1 = !lwe.new_lwe_ciphertext<application_data = <message_type = i16>, plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L1_, key = #key, modulus_chain = #modulus_chain_L5_C1_>
!ct_L2_ = !lwe.new_lwe_ciphertext<application_data = <message_type = tensor<8xi16>>, plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L2_, key = #key, modulus_chain = #modulus_chain_L5_C2_>
!ct_L2_D3_ = !lwe.new_lwe_ciphertext<application_data = <message_type = tensor<8xi16>>, plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L2_D3_, key = #key, modulus_chain = #modulus_chain_L5_C2_>
module attributes {scheme.bgv} {
  func.func @func(%ct: !ct_L2_, %ct_0: !ct_L2_) -> !ct_L0_ {
    %c6 = arith.constant 6 : index
    %cst = arith.constant dense<10> : tensor<8xi16>
    %c7 = arith.constant 7 : index
    %ct_1 = bgv.mul %ct, %ct_0 : (!ct_L2_, !ct_L2_) -> !ct_L2_D3_
    %ct_2 = bgv.relinearize %ct_1 {from_basis = array<i32: 0, 1, 2>, to_basis = array<i32: 0, 1>} : !ct_L2_D3_ -> !ct_L2_
    %pt = lwe.rlwe_encode %cst {encoding = #full_crt_packing_encoding, ring = #ring_Z4295294977_i64_1_x8_} : tensor<8xi16> -> !pt
    %ct_3 = bgv.add_plain %ct_2, %pt : (!ct_L2_, !pt) -> !ct_L2_
    %ct_4 = bgv.rotate %ct_3 {offset = 6 : index} : !ct_L2_
    %ct_5 = bgv.rotate %ct_2 {offset = 7 : index} : !ct_L2_
    %ct_6 = bgv.add %ct_4, %ct_5 : !ct_L2_
    %ct_7 = bgv.add %ct_6, %ct_2 : !ct_L2_
    %ct_8 = bgv.rotate %ct_7 {offset = 6 : index} : !ct_L2_
    %ct_9 = bgv.add %ct_8, %ct_5 : !ct_L2_
    %ct_10 = bgv.add %ct_9, %ct_2 : !ct_L2_
    %ct_11 = bgv.rotate %ct_10 {offset = 6 : index} : !ct_L2_
    %ct_12 = bgv.add %ct_11, %ct_5 : !ct_L2_
    %ct_13 = bgv.add %ct_12, %ct_2 : !ct_L2_
    %ct_14 = bgv.rotate %ct_13 {offset = 7 : index} : !ct_L2_
    %ct_15 = bgv.add %ct_14, %ct_2 : !ct_L2_
    %ct_16 = bgv.modulus_switch %ct_15 {to_ring = #ring_rns_L1_1_x8_} : !ct_L2_ -> !ct_L1_
    %ct_17 = bgv.extract %ct_16, %c7 : (!ct_L1_, index) -> !ct_L1_1
    %ct_18 = bgv.modulus_switch %ct_17 {to_ring = #ring_rns_L0_1_x8_} : !ct_L1_1 -> !ct_L0_
    return %ct_18 : !ct_L0_
  }
  func.func @func__encrypt__arg0(%arg0: tensor<8xi16>, %pk: !pkey_L2_) -> !ct_L2_ {
    %pt = lwe.rlwe_encode %arg0 {encoding = #full_crt_packing_encoding, ring = #ring_Z4295294977_i64_1_x8_} : tensor<8xi16> -> !pt
    %ct = lwe.rlwe_encrypt %pt, %pk : (!pt, !pkey_L2_) -> !ct_L2_
    return %ct : !ct_L2_
  }
  func.func @func__encrypt__arg1(%arg0: tensor<8xi16>, %pk: !pkey_L2_) -> !ct_L2_ {
    %pt = lwe.rlwe_encode %arg0 {encoding = #full_crt_packing_encoding, ring = #ring_Z4295294977_i64_1_x8_} : tensor<8xi16> -> !pt
    %ct = lwe.rlwe_encrypt %pt, %pk : (!pt, !pkey_L2_) -> !ct_L2_
    return %ct : !ct_L2_
  }
  func.func @func__decrypt__result0(%ct: !ct_L0_, %sk: !skey_L0_) -> i16 {
    %pt = lwe.rlwe_decrypt %ct, %sk : (!ct_L0_, !skey_L0_) -> !pt1
    %0 = lwe.rlwe_decode %pt {encoding = #full_crt_packing_encoding, ring = #ring_Z4295294977_i64_1_x8_} : !pt1 -> i16
    return %0 : i16
  }
}

