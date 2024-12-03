// RUN: heir-translate %s --emit-openfhe-pke-pybind --pybind-header-include=foo.h --pybind-module-name=_heir_foo | FileCheck %s

// CHECK: #include <pybind11/pybind11.h>
// CHECK: #include <pybind11/stl.h>
// CHECK: #include "foo.h"
// CHECK: using namespace lbcrypto;
// CHECK: namespace py = pybind11;
// CHECK: void bind_common(py::module &m)
// CHECK: {
// CHECK:     py::class_<PublicKeyImpl<DCRTPoly>, std::shared_ptr<PublicKeyImpl<DCRTPoly>>>(m, "PublicKey")
// CHECK:         .def(py::init<>());
// CHECK:     py::class_<PrivateKeyImpl<DCRTPoly>, std::shared_ptr<PrivateKeyImpl<DCRTPoly>>>(m, "PrivateKey")
// CHECK:         .def(py::init<>());
// CHECK:     py::class_<KeyPair<DCRTPoly>>(m, "KeyPair")
// CHECK:         .def_readwrite("publicKey", &KeyPair<DCRTPoly>::publicKey)
// CHECK:         .def_readwrite("secretKey", &KeyPair<DCRTPoly>::secretKey);
// CHECK:     py::class_<CiphertextImpl<DCRTPoly>, std::shared_ptr<CiphertextImpl<DCRTPoly>>>(m, "Ciphertext")
// CHECK:         .def(py::init<>());
// CHECK:     py::class_<CryptoContextImpl<DCRTPoly>, std::shared_ptr<CryptoContextImpl<DCRTPoly>>>(m, "CryptoContext")
// CHECK:         .def(py::init<>())
// CHECK:         .def("KeyGen", &CryptoContextImpl<DCRTPoly>::KeyGen);
// CHECK: }

// CHECK: PYBIND11_MODULE(_heir_foo, m) {
// CHECK:   bind_common(m);
// CHECK:   m.def("simple_sum", &simple_sum);
// CHECK:   m.def("simple_sum__encrypt", &simple_sum__encrypt);
// CHECK:   m.def("simple_sum__decrypt", &simple_sum__decrypt);
// CHECK: }

#degree_32_poly = #polynomial.int_polynomial<1 + x**32>
#eval_encoding = #lwe.polynomial_evaluation_encoding<cleartext_start = 16, cleartext_bitwidth = 16>
#ring2 = #polynomial.ring<coefficientType=!mod_arith.int<463187969:i32>, polynomialModulus=#degree_32_poly>
#params2 = #lwe.rlwe_params<ring = #ring2>
!tensor_pt_ty = !lwe.rlwe_plaintext<encoding = #eval_encoding, ring = #ring2, underlying_type = tensor<32xi16>>
!scalar_pt_ty = !lwe.rlwe_plaintext<encoding = #eval_encoding, ring = #ring2, underlying_type = i16>
!tensor_ct_ty = !lwe.rlwe_ciphertext<encoding = #eval_encoding, rlwe_params = #params2, underlying_type = tensor<32xi16>>
!scalar_ct_ty = !lwe.rlwe_ciphertext<encoding = #eval_encoding, rlwe_params = #params2, underlying_type = i16>

func.func @simple_sum(%arg0: !openfhe.crypto_context, %arg1: !tensor_ct_ty) -> !scalar_ct_ty {
  %1 = openfhe.rot %arg0, %arg1 { index = 16 } : (!openfhe.crypto_context, !tensor_ct_ty) -> !tensor_ct_ty
  %2 = openfhe.add %arg0, %arg1, %1 : (!openfhe.crypto_context, !tensor_ct_ty, !tensor_ct_ty) -> !tensor_ct_ty
  %4 = openfhe.rot %arg0, %2 { index = 8 } : (!openfhe.crypto_context, !tensor_ct_ty) -> !tensor_ct_ty
  %5 = openfhe.add %arg0, %2, %4 : (!openfhe.crypto_context, !tensor_ct_ty, !tensor_ct_ty) -> !tensor_ct_ty
  %7 = openfhe.rot %arg0, %5 { index = 4 } : (!openfhe.crypto_context, !tensor_ct_ty) -> !tensor_ct_ty
  %8 = openfhe.add %arg0, %5, %7 : (!openfhe.crypto_context, !tensor_ct_ty, !tensor_ct_ty) -> !tensor_ct_ty
  %10 = openfhe.rot %arg0, %8 { index = 2 } : (!openfhe.crypto_context, !tensor_ct_ty) -> !tensor_ct_ty
  %11 = openfhe.add %arg0, %8, %10 : (!openfhe.crypto_context, !tensor_ct_ty, !tensor_ct_ty) -> !tensor_ct_ty
  %13 = openfhe.rot %arg0, %11 { index = 1 } : (!openfhe.crypto_context, !tensor_ct_ty) -> !tensor_ct_ty
  %14 = openfhe.add %arg0, %11, %13 : (!openfhe.crypto_context, !tensor_ct_ty, !tensor_ct_ty) -> !tensor_ct_ty
  %cst = arith.constant dense<[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]> : tensor<32xi16>
  %15 = openfhe.make_packed_plaintext %arg0, %cst : (!openfhe.crypto_context, tensor<32xi16>) -> !tensor_pt_ty
  %16 = openfhe.mul_plain %arg0, %14, %15 : (!openfhe.crypto_context, !tensor_ct_ty, !tensor_pt_ty) -> !tensor_ct_ty
  %18 = openfhe.rot %arg0, %16 { index = 31 } : (!openfhe.crypto_context, !tensor_ct_ty) -> !tensor_ct_ty
  %19 = lwe.reinterpret_underlying_type %18 : !tensor_ct_ty to !scalar_ct_ty
  return %19 : !scalar_ct_ty
}
func.func @simple_sum__encrypt(%arg0: !openfhe.crypto_context, %arg1: tensor<32xi16>, %arg2: !openfhe.public_key) -> !tensor_ct_ty {
  %0 = openfhe.make_packed_plaintext %arg0, %arg1 : (!openfhe.crypto_context, tensor<32xi16>) -> !tensor_pt_ty
  %1 = openfhe.encrypt %arg0, %0, %arg2 : (!openfhe.crypto_context, !tensor_pt_ty, !openfhe.public_key) -> !tensor_ct_ty
  return %1 : !tensor_ct_ty
}
func.func @simple_sum__decrypt(%arg0: !openfhe.crypto_context, %arg1: !scalar_ct_ty, %arg2: !openfhe.private_key) -> i16 {
  %0 = openfhe.decrypt %arg0, %arg1, %arg2 : (!openfhe.crypto_context, !scalar_ct_ty, !openfhe.private_key) -> !scalar_pt_ty
  %1 = lwe.rlwe_decode %0 {encoding = #eval_encoding, ring = #ring2} : !scalar_pt_ty -> i16
  return %1 : i16
}
