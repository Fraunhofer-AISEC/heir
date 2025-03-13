// RUN: heir-opt --verify-diagnostics --split-input-file %s

!Z1095233372161_i64_ = !mod_arith.int<1095233372161 : i64>
!Z65537_i64_ = !mod_arith.int<65537 : i64>

!rns_L0_ = !rns.rns<!Z1095233372161_i64_>
#ring_Z65537_i64_1_x1024_ = #polynomial.ring<coefficientType = !Z65537_i64_, polynomialModulus = <1 + x**1024>>
#ring_rns_L0_1_x1024_ = #polynomial.ring<coefficientType = !rns_L0_, polynomialModulus = <1 + x**1024>>

#full_crt_packing_encoding = #lwe.full_crt_packing_encoding<scaling_factor = 0>
#key = #lwe.key<>

#modulus_chain_L5_C0_ = #lwe.modulus_chain<elements = <1095233372161 : i64, 1032955396097 : i64, 1005037682689 : i64, 998595133441 : i64, 972824936449 : i64, 959939837953 : i64>, current = 0>

#plaintext_space = #lwe.plaintext_space<ring = #ring_Z65537_i64_1_x1024_, encoding = #full_crt_packing_encoding>

#ciphertext_space_L0_D3_ = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x1024_, encryption_type = lsb, size = 3>

!ct = !lwe.new_lwe_ciphertext<application_data = <message_type = i3>, plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L0_D3_, key = #key, modulus_chain = #modulus_chain_L5_C0_>

func.func @test_input_dimension_error(%input: !ct) {
  // expected-error@+1 {{x.dim == 2 does not hold}}
  %out = bgv.rotate_cols  %input { offset = 4 }  : !ct
  return
}

// -----

!Z1095233372161_i64_ = !mod_arith.int<1095233372161 : i64>
!Z65537_i64_ = !mod_arith.int<65537 : i64>

!rns_L0_ = !rns.rns<!Z1095233372161_i64_>
#ring_Z65537_i64_1_x1024_ = #polynomial.ring<coefficientType = !Z65537_i64_, polynomialModulus = <1 + x**1024>>
#ring_rns_L0_1_x1024_ = #polynomial.ring<coefficientType = !rns_L0_, polynomialModulus = <1 + x**1024>>

#full_crt_packing_encoding = #lwe.full_crt_packing_encoding<scaling_factor = 0>
#key = #lwe.key<>

#modulus_chain_L5_C0_ = #lwe.modulus_chain<elements = <1095233372161 : i64, 1032955396097 : i64, 1005037682689 : i64, 998595133441 : i64, 972824936449 : i64, 959939837953 : i64>, current = 0>

#plaintext_space = #lwe.plaintext_space<ring = #ring_Z65537_i64_1_x1024_, encoding = #full_crt_packing_encoding>

#ciphertext_space_L0_ = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x1024_, encryption_type = lsb>

!ct = !lwe.new_lwe_ciphertext<application_data = <message_type = i3>, plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L0_, key = #key, modulus_chain = #modulus_chain_L5_C0_>


func.func @test_extract_verification_failure(%input: !ct) -> !ct {
  %offset = arith.constant 4 : index
  // expected-error@+1 {{'bgv.extract' op input RLWE ciphertext type must have a ranked tensor as its underlying_type, but found 'i3'}}
  %ext = bgv.extract %input, %offset : (!ct, index) -> !ct
  return %ext : !ct
}

// -----

!Z1095233372161_i64_ = !mod_arith.int<1095233372161 : i64>
!Z65537_i64_ = !mod_arith.int<65537 : i64>

!rns_L0_ = !rns.rns<!Z1095233372161_i64_>
#ring_Z65537_i64_1_x1024_ = #polynomial.ring<coefficientType = !Z65537_i64_, polynomialModulus = <1 + x**1024>>
#ring_rns_L0_1_x1024_ = #polynomial.ring<coefficientType = !rns_L0_, polynomialModulus = <1 + x**1024>>

#full_crt_packing_encoding = #lwe.full_crt_packing_encoding<scaling_factor = 0>
#key = #lwe.key<>

#modulus_chain_L5_C0_ = #lwe.modulus_chain<elements = <1095233372161 : i64, 1032955396097 : i64, 1005037682689 : i64, 998595133441 : i64, 972824936449 : i64, 959939837953 : i64>, current = 0>

#plaintext_space = #lwe.plaintext_space<ring = #ring_Z65537_i64_1_x1024_, encoding = #full_crt_packing_encoding>

#ciphertext_space_L0_ = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x1024_, encryption_type = lsb>

!ct_tensor = !lwe.new_lwe_ciphertext<application_data = <message_type = tensor<64xi16>>, plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L0_, key = #key, modulus_chain = #modulus_chain_L5_C0_>
!ct_scalar = !lwe.new_lwe_ciphertext<application_data = <message_type = i3>, plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L0_, key = #key, modulus_chain = #modulus_chain_L5_C0_>

func.func @test_extract_verification_failure(%input: !ct_tensor) -> !ct_scalar {
  %offset = arith.constant 4 : index
  // expected-error@+1 {{'bgv.extract' op output RLWE ciphertext's underlying_type must be the element type of the input ciphertext's underlying tensor type, but found tensor type 'tensor<64xi16>' and output type 'i3'}}
  %ext = bgv.extract %input, %offset : (!ct_tensor, index) -> !ct_scalar
  return %ext : !ct_scalar
}

// -----

!Z17592186175489_i64_ = !mod_arith.int<17592186175489 : i64>
!Z33832961_i64_ = !mod_arith.int<33832961 : i64>
!Z65537_i64_ = !mod_arith.int<65537 : i64>
// note the scaling factor is 2
// after mul it should be 4
#full_crt_packing_encoding = #lwe.full_crt_packing_encoding<scaling_factor = 2>
#key = #lwe.key<>
#modulus_chain_L1_C0_ = #lwe.modulus_chain<elements = <33832961 : i64, 17592186175489 : i64>, current = 0>
#modulus_chain_L1_C1_ = #lwe.modulus_chain<elements = <33832961 : i64, 17592186175489 : i64>, current = 1>
!rns_L0_ = !rns.rns<!Z33832961_i64_>
!rns_L1_ = !rns.rns<!Z33832961_i64_, !Z17592186175489_i64_>
#ring_Z65537_i64_1_x1024_ = #polynomial.ring<coefficientType = !Z65537_i64_, polynomialModulus = <1 + x**1024>>
#ring_rns_L0_1_x1024_ = #polynomial.ring<coefficientType = !rns_L0_, polynomialModulus = <1 + x**1024>>
#ring_rns_L1_1_x1024_ = #polynomial.ring<coefficientType = !rns_L1_, polynomialModulus = <1 + x**1024>>
#ciphertext_space_L0_ = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x1024_, encryption_type = lsb>
#ciphertext_space_L1_ = #lwe.ciphertext_space<ring = #ring_rns_L1_1_x1024_, encryption_type = lsb>
#ciphertext_space_L1_D3_ = #lwe.ciphertext_space<ring = #ring_rns_L1_1_x1024_, encryption_type = lsb, size = 3>
!ct_L0_ = !lwe.new_lwe_ciphertext<application_data = <message_type = i16>, plaintext_space = <ring = #ring_Z65537_i64_1_x1024_, encoding = #full_crt_packing_encoding>, ciphertext_space = #ciphertext_space_L0_, key = #key, modulus_chain = #modulus_chain_L1_C0_>
!ct_L1_ = !lwe.new_lwe_ciphertext<application_data = <message_type = i16>, plaintext_space = <ring = #ring_Z65537_i64_1_x1024_, encoding = #full_crt_packing_encoding>, ciphertext_space = #ciphertext_space_L1_, key = #key, modulus_chain = #modulus_chain_L1_C1_>
!ct_L1_D3_ = !lwe.new_lwe_ciphertext<application_data = <message_type = i16>, plaintext_space = <ring = #ring_Z65537_i64_1_x1024_, encoding = #full_crt_packing_encoding>, ciphertext_space = #ciphertext_space_L1_D3_, key = #key, modulus_chain = #modulus_chain_L1_C1_>
module attributes {bgv.schemeParam = #bgv.scheme_param<logN = 13, Q = [33832961, 17592186175489], P = [17592186273793], plaintextModulus = 65537>, scheme.bgv} {
  func.func @mul(%ct: !ct_L1_) -> !ct_L0_ {
    // expected-error@+1 {{'bgv.mul' op output plaintext space does not match}}
    %ct_0 = bgv.mul %ct, %ct : (!ct_L1_, !ct_L1_) -> !ct_L1_D3_
    %ct_1 = bgv.relinearize %ct_0 {from_basis = array<i32: 0, 1, 2>, to_basis = array<i32: 0, 1>} : !ct_L1_D3_ -> !ct_L1_
    %ct_2 = bgv.modulus_switch %ct_1 {to_ring = #ring_rns_L0_1_x1024_} : !ct_L1_ -> !ct_L0_
    return %ct_2 : !ct_L0_
  }
}

// -----

!Z17592186175489_i64_ = !mod_arith.int<17592186175489 : i64>
!Z33832961_i64_ = !mod_arith.int<33832961 : i64>
!Z65537_i64_ = !mod_arith.int<65537 : i64>
// note the scaling factor is 1
// after mul it should be 1, it is ok
// but after modulus switching it should be 1 * q^{-1} % t
#full_crt_packing_encoding = #lwe.full_crt_packing_encoding<scaling_factor = 1>
#key = #lwe.key<>
#modulus_chain_L1_C0_ = #lwe.modulus_chain<elements = <33832961 : i64, 17592186175489 : i64>, current = 0>
#modulus_chain_L1_C1_ = #lwe.modulus_chain<elements = <33832961 : i64, 17592186175489 : i64>, current = 1>
!rns_L0_ = !rns.rns<!Z33832961_i64_>
!rns_L1_ = !rns.rns<!Z33832961_i64_, !Z17592186175489_i64_>
#ring_Z65537_i64_1_x1024_ = #polynomial.ring<coefficientType = !Z65537_i64_, polynomialModulus = <1 + x**1024>>
#ring_rns_L0_1_x1024_ = #polynomial.ring<coefficientType = !rns_L0_, polynomialModulus = <1 + x**1024>>
#ring_rns_L1_1_x1024_ = #polynomial.ring<coefficientType = !rns_L1_, polynomialModulus = <1 + x**1024>>
#ciphertext_space_L0_ = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x1024_, encryption_type = lsb>
#ciphertext_space_L1_ = #lwe.ciphertext_space<ring = #ring_rns_L1_1_x1024_, encryption_type = lsb>
#ciphertext_space_L1_D3_ = #lwe.ciphertext_space<ring = #ring_rns_L1_1_x1024_, encryption_type = lsb, size = 3>
!ct_L0_ = !lwe.new_lwe_ciphertext<application_data = <message_type = i16>, plaintext_space = <ring = #ring_Z65537_i64_1_x1024_, encoding = #full_crt_packing_encoding>, ciphertext_space = #ciphertext_space_L0_, key = #key, modulus_chain = #modulus_chain_L1_C0_>
!ct_L1_ = !lwe.new_lwe_ciphertext<application_data = <message_type = i16>, plaintext_space = <ring = #ring_Z65537_i64_1_x1024_, encoding = #full_crt_packing_encoding>, ciphertext_space = #ciphertext_space_L1_, key = #key, modulus_chain = #modulus_chain_L1_C1_>
!ct_L1_D3_ = !lwe.new_lwe_ciphertext<application_data = <message_type = i16>, plaintext_space = <ring = #ring_Z65537_i64_1_x1024_, encoding = #full_crt_packing_encoding>, ciphertext_space = #ciphertext_space_L1_D3_, key = #key, modulus_chain = #modulus_chain_L1_C1_>
module attributes {bgv.schemeParam = #bgv.scheme_param<logN = 13, Q = [33832961, 17592186175489], P = [17592186273793], plaintextModulus = 65537>, scheme.bgv} {
  func.func @mul(%ct: !ct_L1_) -> !ct_L0_ {
    %ct_0 = bgv.mul %ct, %ct : (!ct_L1_, !ct_L1_) -> !ct_L1_D3_
    %ct_1 = bgv.relinearize %ct_0 {from_basis = array<i32: 0, 1, 2>, to_basis = array<i32: 0, 1>} : !ct_L1_D3_ -> !ct_L1_
    // expected-error@+1 {{'bgv.modulus_switch' op output plaintext space does not match}}
    %ct_2 = bgv.modulus_switch %ct_1 {to_ring = #ring_rns_L0_1_x1024_} : !ct_L1_ -> !ct_L0_
    return %ct_2 : !ct_L0_
  }
}
