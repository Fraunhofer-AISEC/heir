// RUN: heir-opt -forward-insert-to-extract %s | FileCheck %s


!cc = !openfhe.crypto_context

!Z1095233372161_i64_ = !mod_arith.int<1095233372161 : i64>
!Z65537_i64_ = !mod_arith.int<65537 : i64>
#key = #lwe.key<>
#modulus_chain_L5_C0_ = #lwe.modulus_chain<elements = <1095233372161 : i64, 1032955396097 : i64, 1005037682689 : i64, 998595133441 : i64, 972824936449 : i64, 959939837953 : i64>, current = 0>
!rns_L0_ = !rns.rns<!Z1095233372161_i64_>
#ring_rns_L0_1_x16_ = #polynomial.ring<coefficientType = !rns_L0_, polynomialModulus = <1 + x**16>>
#ring_Z65537_i64_1_x16_ = #polynomial.ring<coefficientType = !Z65537_i64_, polynomialModulus = <1 + x**16>>
#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 1024>
#plaintext_space = #lwe.plaintext_space<ring = #ring_Z65537_i64_1_x16_, encoding = #inverse_canonical_encoding>
#ciphertext_space_L0_ = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x16_, encryption_type = lsb>
!pt = !lwe.new_lwe_plaintext<application_data = <message_type = f32>, plaintext_space = #plaintext_space>
!ct = !lwe.new_lwe_ciphertext<application_data = <message_type = f32>, plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L0_, key = #key, modulus_chain = #modulus_chain_L5_C0_>


//  CHECK-LABEL: @successful_forwarding
//  CHECK-SAME:  (%[[ARG0:.*]]: !cc,


func.func @successful_forwarding(%arg0: !cc, %arg1: tensor<1x16x!ct>, %arg2: tensor<1x16x!ct>, %arg3: tensor<16xf64>, %arg4: tensor<16xf64>) -> tensor<1x16x!ct> {

  // CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  // CHECK-NEXT: %[[C1:.*]] = arith.constant 1 : index
  %c1 = arith.constant 1 : index

  //  CHECK-NEXT: %[[EXTRACTED:.*]] = tensor.extract
  %extracted = tensor.extract %arg1[%c0, %c0] : tensor<1x16x!ct>
  //  CHECK-NEXT: %[[EXTRACTED0:.*]] = tensor.extract
  %extracted_0 = tensor.extract %arg2[%c0, %c0] : tensor<1x16x!ct>
  //  CHECK-NEXT: %[[VAL0:.*]] = openfhe.make_ckks_packed_plaintext %[[ARG0]]
  %0 = openfhe.make_ckks_packed_plaintext %arg0, %arg3 : (!cc, tensor<16xf64>) -> !pt
  //  CHECK-NEXT: %[[VAL1:.*]] = openfhe.mul_plain %[[ARG0]], %[[EXTRACTED]], %[[VAL0]]
  %1 = openfhe.mul_plain %arg0, %extracted, %0 : (!cc, !ct, !pt) -> !ct
  //  CHECK-NEXT: %[[VAL2:.*]] = openfhe.add %[[ARG0]], %[[EXTRACTED0]], %[[VAL1]]
  %2 = openfhe.add %arg0, %extracted_0, %1 : (!cc, !ct, !ct) -> !ct

  //  CHECK-NEXT: %[[INSERTED0:.*]] = tensor.insert %[[VAL2]]
  %inserted = tensor.insert %2 into %arg2[%c0, %c0] : tensor<1x16x!ct>

  //  CHECK-NEXT: %[[EXTRACTED1:.*]] = tensor.extract
  %extracted_1 = tensor.extract %arg1[%c0, %c1] : tensor<1x16x!ct>
  //  CHECK-NOT: tensor.extract %[[INSERTED0]]
  %extracted_2 = tensor.extract %inserted[%c0, %c0] : tensor<1x16x!ct>
  //  CHECK-NEXT: %[[VAL3:.*]] = openfhe.make_ckks_packed_plaintext
  %3 = openfhe.make_ckks_packed_plaintext %arg0, %arg4 : (!cc, tensor<16xf64>) -> !pt
  //  CHECK-NEXT: %[[VAL4:.*]] = openfhe.mul_plain
  %4 = openfhe.mul_plain %arg0, %extracted_1, %3 : (!cc, !ct, !pt) -> !ct
  //  CHECK-NEXT: %[[VAL5:.*]] = openfhe.add
  %5 = openfhe.add %arg0, %extracted_2, %4 : (!cc, !ct, !ct) -> !ct
  //  CHECK-NEXT: %[[INSERTED1:.*]] = tensor.insert
  %inserted_3 = tensor.insert %5 into %inserted[%c0, %c0] : tensor<1x16x!ct>
  //  CHECK-NEXT: return %[[INSERTED1]]
  return %inserted_3 : tensor<1x16x!ct>
}


//hits def == nullptr
//  CHECK-LABEL: @forward_from_func_arg
//  CHECK-SAME:  (%[[ARG0:.*]]: !cc,

func.func @forward_from_func_arg(%arg0: !cc, %arg1: tensor<1x16x!ct>, %arg2: tensor<1x16x!ct>)-> !ct {
  // CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  //  CHECK-NEXT: %[[EXTRACTED:.*]] = tensor.extract
  %extracted = tensor.extract %arg1[%c0, %c0] : tensor<1x16x!ct>

  return %extracted : !ct
}

//  CHECK-LABEL: @forwarding_with_an_insert_in_between
//  CHECK-SAME:  (%[[ARG0:.*]]: !cc,

func.func @forwarding_with_an_insert_in_between(%arg0: !cc, %arg1: tensor<1x16x!ct>, %arg2: tensor<1x16x!ct>, %arg3: tensor<16xf64> )-> !ct {

  // CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : index
  %c0 = arith.constant 0 : index

  //  CHECK-NEXT: %[[EXTRACTED:.*]] = tensor.extract
  %extracted = tensor.extract %arg1[%c0, %c0] : tensor<1x16x!ct>
  //  CHECK-NEXT: %[[EXTRACTED0:.*]] = tensor.extract
  %extracted_0 = tensor.extract %arg2[%c0, %c0] : tensor<1x16x!ct>
  //  CHECK-NEXT: %[[VAL0:.*]] = openfhe.make_ckks_packed_plaintext %[[ARG0]]
  %0 = openfhe.make_ckks_packed_plaintext %arg0, %arg3 : (!cc, tensor<16xf64>) -> !pt
  //  CHECK-NEXT: %[[VAL1:.*]] = openfhe.mul_plain %[[ARG0]], %[[EXTRACTED]], %[[VAL0]]
  %1 = openfhe.mul_plain %arg0, %extracted, %0 : (!cc, !ct, !pt) -> !ct
  //  CHECK-NEXT: %[[VAL2:.*]] = openfhe.add %[[ARG0]], %[[EXTRACTED0]], %[[VAL1]]
  %2 = openfhe.add %arg0, %extracted_0, %1 : (!cc, !ct, !ct) -> !ct
  //  CHECK-NEXT: %[[VALA2:.*]] = openfhe.add %[[ARG0]], %[[EXTRACTED0]], %[[VAL2]]
  %a2 = openfhe.add %arg0, %extracted_0, %2 : (!cc, !ct, !ct) -> !ct
  //  CHECK-NOT: tensor.insert %[[VAL2]]
  %inserted = tensor.insert %2 into %arg2[%c0, %c0] : tensor<1x16x!ct>
  //  CHECK-NOT: tensor.insert %[[VALA2]]
  %inserted_1 = tensor.insert %a2 into %arg1[%c0, %c0] : tensor<1x16x!ct>

  //  CHECK-NOT: tensor.extract
  %extracted_2 = tensor.extract %inserted_1[%c0, %c0] : tensor<1x16x!ct>
  // CHECK: return %[[VALA2]]
  return %extracted_2 : !ct
}

//  CHECK-LABEL: @forwarding_with_an_operation_in_between
//  CHECK-SAME:  (%[[ARG0:.*]]: !cc,

func.func @forwarding_with_an_operation_in_between(%arg0: !cc, %arg1: tensor<1x16x!ct>, %arg2: tensor<1x16x!ct>, %arg3: tensor<16xf64>, %arg4: i1 )-> !ct {

  // CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : index
  %c0 = arith.constant 0 : index

  //  CHECK-NEXT: %[[EXTRACTED:.*]] = tensor.extract
  %extracted = tensor.extract %arg1[%c0, %c0] : tensor<1x16x!ct>
  //  CHECK-NEXT: %[[EXTRACTED0:.*]] = tensor.extract
  %extracted_0 = tensor.extract %arg2[%c0, %c0] : tensor<1x16x!ct>
  //  CHECK-NEXT: %[[VAL0:.*]] = openfhe.make_ckks_packed_plaintext %[[ARG0]]
  %0 = openfhe.make_ckks_packed_plaintext %arg0, %arg3 : (!cc, tensor<16xf64>) -> !pt
  //  CHECK-NEXT: %[[VAL1:.*]] = openfhe.mul_plain %[[ARG0]], %[[EXTRACTED]], %[[VAL0]]
  %1 = openfhe.mul_plain %arg0, %extracted, %0 : (!cc, !ct, !pt) -> !ct
  //  CHECK-NEXT: %[[VAL2:.*]] = openfhe.add %[[ARG0]], %[[EXTRACTED0]], %[[VAL1]]
  %2 = openfhe.add %arg0, %extracted_0, %1 : (!cc, !ct, !ct) -> !ct

  //  CHECK-NOT: %[[INSERTED0:.*]] = tensor.insert %[[VAL2]]
  %inserted = tensor.insert %2 into %arg2[%c0, %c0] : tensor<1x16x!ct>

  scf.if %arg4 {
    //  CHECK-NOT: %[[VALa2:.*]] = openfhe.add %[[ARG0]], %[[EXTRACTED0]], %[[VAL2]]
    %a2 = openfhe.add %arg0, %extracted_0, %2 : (!cc, !ct, !ct) -> !ct
    //  CHECK-NOT: tensor.insert %[[VAL1]]
    %inserted_1 = tensor.insert %a2 into %arg2[%c0, %c0] : tensor<1x16x!ct>
  }
    //  CHECK-NOT: tensor.extract
  %extracted_2 = tensor.extract %inserted[%c0, %c0] : tensor<1x16x!ct>
  return %extracted_2 : !ct
}


//  CHECK-LABEL: @two_extracts_both_forwarded
//  CHECK-SAME:  (%[[ARG0:.*]]: !cc,

func.func @two_extracts_both_forwarded(%arg0: !cc, %arg1: tensor<1x16x!ct>, %arg2: tensor<1x16x!ct>, %arg3: tensor<16xf64>) -> !ct {

  // CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : index
  %c0 = arith.constant 0 : index

  //  CHECK-NEXT: %[[EXTRACTED:.*]] = tensor.extract
  %extracted = tensor.extract %arg1[%c0, %c0] : tensor<1x16x!ct>
  //  CHECK-NEXT: %[[EXTRACTED0:.*]] = tensor.extract
  %extracted_0 = tensor.extract %arg2[%c0, %c0] : tensor<1x16x!ct>
  //  CHECK-NEXT: %[[VAL0:.*]] = openfhe.make_ckks_packed_plaintext %[[ARG0]]
  %0 = openfhe.make_ckks_packed_plaintext %arg0, %arg3 : (!cc, tensor<16xf64>) -> !pt
  //  CHECK-NEXT: %[[VAL1:.*]] = openfhe.mul_plain %[[ARG0]], %[[EXTRACTED]], %[[VAL0]]
  %1 = openfhe.mul_plain %arg0, %extracted, %0 : (!cc, !ct, !pt) -> !ct
  //  CHECK-NEXT: %[[VAL2:.*]] = openfhe.add %[[ARG0]], %[[EXTRACTED0]], %[[VAL1]]
  %2 = openfhe.add %arg0, %extracted_0, %1 : (!cc, !ct, !ct) -> !ct

  %inserted = tensor.insert %2 into %arg2[%c0, %c0] : tensor<1x16x!ct>

  //  CHECK-NOT: tensor.extract
  %extracted_1 = tensor.extract %inserted[%c0, %c0] : tensor<1x16x!ct>
  //  CHECK-NOT: tensor.extract
  %extracted_2 = tensor.extract %inserted[%c0, %c0] : tensor<1x16x!ct>
  // CHECK: openfhe.add %[[ARG0]], %[[VAL2]], %[[VAL2]]
  %3 = openfhe.add %arg0, %extracted_1, %extracted_2 : (!cc, !ct, !ct) -> !ct
  return %3: !ct
}
