// RUN: heir-translate %s --emit-openfhe-pke | FileCheck %s

!cc = !openfhe.crypto_context

!Z1095233372161_i64_ = !mod_arith.int<1095233372161 : i64>
!Z65537_i64_ = !mod_arith.int<65537 : i64>
#key = #lwe.key<>
#modulus_chain_L5_C0_ = #lwe.modulus_chain<elements = <1095233372161 : i64, 1032955396097 : i64, 1005037682689 : i64, 998595133441 : i64, 972824936449 : i64, 959939837953 : i64>, current = 0>
!rns_L0_ = !rns.rns<!Z1095233372161_i64_>
#ring_rns_L0_1_x8_ = #polynomial.ring<coefficientType = !rns_L0_, polynomialModulus = <1 + x**8>>
#ring_Z65537_i64_1_x8_ = #polynomial.ring<coefficientType = !Z65537_i64_, polynomialModulus = <1 + x**8>>
#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 1024>
#plaintext_space = #lwe.plaintext_space<ring = #ring_Z65537_i64_1_x8_, encoding = #inverse_canonical_encoding>
#ciphertext_space_L0_ = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x8_, encryption_type = lsb>
!pt = !lwe.new_lwe_plaintext<application_data = <message_type = f16>, plaintext_space = #plaintext_space>
!ct = !lwe.new_lwe_ciphertext<application_data = <message_type = f16>, plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L0_, key = #key, modulus_chain = #modulus_chain_L5_C0_>

// [(input1 + input2) - input3] * input4
module attributes {scheme.bgv} {
  func.func @test_ciphertext_plaintext_ops(%cc : !cc, %input1 : !ct, %input2 : !pt, %input3 : !pt, %input4 : !pt) -> !ct {
    %add_res = openfhe.add_plain %cc, %input1, %input2 : (!cc, !ct, !pt) -> !ct
    %sub_res = openfhe.sub_plain %cc, %add_res, %input3 : (!cc, !ct, !pt) -> !ct
    %mul_res = openfhe.mul_plain %cc, %sub_res, %input4 : (!cc, !ct, !pt) -> !ct
    return %mul_res : !ct
  }
}
