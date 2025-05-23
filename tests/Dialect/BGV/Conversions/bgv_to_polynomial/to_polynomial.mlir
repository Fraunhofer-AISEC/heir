// RUN: heir-opt --mlir-print-local-scope --bgv-to-lwe --lwe-to-polynomial %s > %t
// RUN: FileCheck %s < %t

#encoding = #lwe.polynomial_evaluation_encoding<cleartext_start=30, cleartext_bitwidth=3>

#my_poly = #polynomial.int_polynomial<1 + x**1024>
#ring = #polynomial.ring<coefficientType=!mod_arith.int<33538049:i32>, polynomialModulus=#my_poly>
#params = #lwe.rlwe_params<dimension=2, ring=#ring>
#params1 = #lwe.rlwe_params<dimension=3, ring=#ring>

!ct1 = !lwe.rlwe_ciphertext<encoding=#encoding, rlwe_params=#params, underlying_type=i3>
!ct2 = !lwe.rlwe_ciphertext<encoding=#encoding, rlwe_params=#params1, underlying_type=i3>

!pt1 = !lwe.rlwe_plaintext<encoding=#encoding, ring=#ring, underlying_type=i3>

// CHECK: module
module {
  // CHECK: func.func @test_fn([[X:%.+]]: [[T:tensor<2x!polynomial.*33538049.*]]) -> [[T]] {
  func.func @test_fn(%x : !ct1) -> !ct1 {
    // CHECK: return [[X]] : [[T]]
    return %x : !ct1
  }


  // CHECK: func.func @test_bin_ops([[X:%.+]]: [[T:tensor<2x!polynomial.*33538049.*]], [[Y:%.+]]: [[T]], [[A:%.+]]: [[PT:tensor<2x!polynomial.*33538049.*]])
  func.func @test_bin_ops(%x : !ct1, %y : !ct1, %z : !pt1) -> (!ct1, !ct1, !ct1, !ct1, !ct1, !ct2, !ct1) {
    // CHECK: polynomial.add [[X]], [[Y]] : [[T]]
    %add = bgv.add %x, %y  : !ct1
    // CHECK: polynomial.add [[X]], [[A]] : [[T]]
    %add_plain = bgv.add_plain %x, %z : (!ct1, !pt1) -> !ct1
    // CHECK: polynomial.sub [[X]], [[Y]] : [[T]]
    %sub = bgv.sub %x, %y  : !ct1
    // CHECK: polynomial.sub [[X]], [[A]] : [[T]]
    %sub_plain = bgv.sub_plain %x %z : !ct1 (!ct1, !pt1) -> !ct1
    // CHECK: [[C:%.+]] = mod_arith.constant -1 : [[I:.+]]
    // CHECK: polynomial.mul_scalar [[X]], [[C]] : [[T]], [[I]]
    %negate = bgv.negate %x  : !ct1

    // CHECK: [[I0:%.+]] = arith.constant 0 : index
    // CHECK: [[I1:%.+]] = arith.constant 1 : index
    // CHECK: [[X0:%.+]] = tensor.extract [[X]][[[I0]]] : [[T]]
    // CHECK: [[X1:%.+]] = tensor.extract [[X]][[[I1]]] : [[T]]
    // CHECK: [[Y0:%.+]] = tensor.extract [[Y]][[[I0]]] : [[T]]
    // CHECK: [[Y1:%.+]] = tensor.extract [[Y]][[[I1]]] : [[T]]
    // CHECK: [[Z0:%.+]] = polynomial.mul [[X0]], [[Y0]] : [[P:!polynomial.*33538049.*]]
    // CHECK: [[X0Y1:%.+]] = polynomial.mul [[X0]], [[Y1]] : [[P]]
    // CHECK: [[X1Y0:%.+]] = polynomial.mul [[X1]], [[Y0]] : [[P]]
    // CHECK: [[Z1:%.+]] = polynomial.add [[X0Y1]], [[X1Y0]] : [[P]]
    // CHECK: [[Z2:%.+]] = polynomial.mul [[X1]], [[Y1]] : [[P]]
    // CHECK: [[Z:%.+]] = tensor.from_elements [[Z0]], [[Z1]], [[Z2]] : tensor<3x[[P]]>
    %mul = bgv.mul %x, %y  : (!ct1, !ct1) -> !ct2

    // CHECK: [[I0:%.+]] = arith.constant 0 : index
    // CHECK: [[I1:%.+]] = arith.constant 1 : index
    // CHECK: [[X0:%.+]] = tensor.extract [[X]][[[I0]]] : [[T]]
    // CHECK: [[X1:%.+]] = tensor.extract [[X]][[[I1]]] : [[T]]
    // CHECK: [[Y0:%.+]] = tensor.extract [[Y]][[[I0]]] : [[PT]]
    // CHECK: [[Z0:%.+]] = polynomial.mul [[X0]], [[Y0]] : [[P:!polynomial.*33538049.*]]
    // CHECK: [[Z1:%.+]] = polynomial.mul [[X1]], [[Y0]] : [[P]]
    // CHECK: [[Z:%.+]] = tensor.from_elements [[Z0]], [[Z1]] : tensor<2x[[P]]>
    %mul_plain = bgv.mul_plain %x, %z : (!ct1, !pt1) -> !ct1
    return %add, %add_plain, %sub, %sub_plain, %negate, %mul, %mul_plain: !ct1, !ct1, !ct1, !ct1, !ct1, !ct2, !ct1
  }
}
