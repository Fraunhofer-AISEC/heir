#unspecified_bit_field_encoding = #lwe.unspecified_bit_field_encoding<cleartext_bitwidth = 3>
module attributes {scheme.cggi, tf_saved_model.semantics} {
  func.func @main(%arg0: memref<1x1x8x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>, %arg1: !jaxite.server_key_set, %arg2: !jaxite.params) -> memref<1x1x32x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>> {
    %c31 = arith.constant 31 : index
    %c30 = arith.constant 30 : index
    %c29 = arith.constant 29 : index
    %c28 = arith.constant 28 : index
    %c27 = arith.constant 27 : index
    %c26 = arith.constant 26 : index
    %c25 = arith.constant 25 : index
    %c24 = arith.constant 24 : index
    %c23 = arith.constant 23 : index
    %c22 = arith.constant 22 : index
    %c21 = arith.constant 21 : index
    %c20 = arith.constant 20 : index
    %c19 = arith.constant 19 : index
    %c18 = arith.constant 18 : index
    %c17 = arith.constant 17 : index
    %c16 = arith.constant 16 : index
    %c15 = arith.constant 15 : index
    %c14 = arith.constant 14 : index
    %c13 = arith.constant 13 : index
    %c12 = arith.constant 12 : index
    %c11 = arith.constant 11 : index
    %c10 = arith.constant 10 : index
    %c9 = arith.constant 9 : index
    %c8 = arith.constant 8 : index
    %false = arith.constant false
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %c6 = arith.constant 6 : index
    %c7 = arith.constant 7 : index
    %c6_i8 = arith.constant 6 : i8
    %c120_i8 = arith.constant 120 : i8
    %c-128_i8 = arith.constant -128 : i8
    %c9_i8 = arith.constant 9 : i8
    %c4_i8 = arith.constant 4 : i8
    %c1_i8 = arith.constant 1 : i8
    %0 = memref.load %arg0[%c0, %c0, %c1] : memref<1x1x8x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    %1 = memref.load %arg0[%c0, %c0, %c0] : memref<1x1x8x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    %2 = jaxite.constant %false, %arg2 : (i1, !jaxite.params) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %3 = jaxite.lut3 %1, %0, %2, %c6_i8, %arg1, %arg2 : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, i8, !jaxite.server_key_set, !jaxite.params) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %4 = memref.load %arg0[%c0, %c0, %c2] : memref<1x1x8x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    %5 = jaxite.lut3 %1, %0, %4, %c120_i8, %arg1, %arg2 : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, i8, !jaxite.server_key_set, !jaxite.params) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %6 = jaxite.lut3 %1, %0, %4, %c-128_i8, %arg1, %arg2 : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, i8, !jaxite.server_key_set, !jaxite.params) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %7 = memref.load %arg0[%c0, %c0, %c3] : memref<1x1x8x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    %8 = jaxite.lut3 %6, %7, %2, %c6_i8, %arg1, %arg2 : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, i8, !jaxite.server_key_set, !jaxite.params) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %9 = memref.load %arg0[%c0, %c0, %c4] : memref<1x1x8x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    %10 = jaxite.lut3 %6, %7, %9, %c120_i8, %arg1, %arg2 : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, i8, !jaxite.server_key_set, !jaxite.params) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %11 = jaxite.lut3 %6, %7, %9, %c-128_i8, %arg1, %arg2 : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, i8, !jaxite.server_key_set, !jaxite.params) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %12 = memref.load %arg0[%c0, %c0, %c5] : memref<1x1x8x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    %13 = jaxite.lut3 %11, %12, %2, %c6_i8, %arg1, %arg2 : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, i8, !jaxite.server_key_set, !jaxite.params) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %14 = memref.load %arg0[%c0, %c0, %c6] : memref<1x1x8x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    %15 = jaxite.lut3 %11, %12, %14, %c120_i8, %arg1, %arg2 : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, i8, !jaxite.server_key_set, !jaxite.params) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %16 = jaxite.lut3 %11, %12, %14, %c-128_i8, %arg1, %arg2 : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, i8, !jaxite.server_key_set, !jaxite.params) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %17 = memref.load %arg0[%c0, %c0, %c7] : memref<1x1x8x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    %18 = jaxite.lut3 %16, %17, %2, %c9_i8, %arg1, %arg2 : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, i8, !jaxite.server_key_set, !jaxite.params) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %19 = jaxite.lut3 %17, %16, %2, %c4_i8, %arg1, %arg2 : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, i8, !jaxite.server_key_set, !jaxite.params) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %20 = jaxite.lut3 %1, %2, %2, %c1_i8, %arg1, %arg2 : (!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>, i8, !jaxite.server_key_set, !jaxite.params) -> !lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>
    %alloc = memref.alloc() : memref<1x1x32x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    memref.store %20, %alloc[%c0, %c0, %c0] : memref<1x1x32x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    memref.store %3, %alloc[%c0, %c0, %c1] : memref<1x1x32x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    memref.store %5, %alloc[%c0, %c0, %c2] : memref<1x1x32x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    memref.store %8, %alloc[%c0, %c0, %c3] : memref<1x1x32x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    memref.store %10, %alloc[%c0, %c0, %c4] : memref<1x1x32x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    memref.store %13, %alloc[%c0, %c0, %c5] : memref<1x1x32x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    memref.store %15, %alloc[%c0, %c0, %c6] : memref<1x1x32x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    memref.store %18, %alloc[%c0, %c0, %c7] : memref<1x1x32x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    memref.store %19, %alloc[%c0, %c0, %c8] : memref<1x1x32x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    memref.store %2, %alloc[%c0, %c0, %c9] : memref<1x1x32x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    memref.store %2, %alloc[%c0, %c0, %c10] : memref<1x1x32x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    memref.store %2, %alloc[%c0, %c0, %c11] : memref<1x1x32x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    memref.store %2, %alloc[%c0, %c0, %c12] : memref<1x1x32x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    memref.store %2, %alloc[%c0, %c0, %c13] : memref<1x1x32x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    memref.store %2, %alloc[%c0, %c0, %c14] : memref<1x1x32x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    memref.store %2, %alloc[%c0, %c0, %c15] : memref<1x1x32x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    memref.store %2, %alloc[%c0, %c0, %c16] : memref<1x1x32x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    memref.store %2, %alloc[%c0, %c0, %c17] : memref<1x1x32x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    memref.store %2, %alloc[%c0, %c0, %c18] : memref<1x1x32x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    memref.store %2, %alloc[%c0, %c0, %c19] : memref<1x1x32x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    memref.store %2, %alloc[%c0, %c0, %c20] : memref<1x1x32x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    memref.store %2, %alloc[%c0, %c0, %c21] : memref<1x1x32x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    memref.store %2, %alloc[%c0, %c0, %c22] : memref<1x1x32x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    memref.store %2, %alloc[%c0, %c0, %c23] : memref<1x1x32x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    memref.store %2, %alloc[%c0, %c0, %c24] : memref<1x1x32x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    memref.store %2, %alloc[%c0, %c0, %c25] : memref<1x1x32x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    memref.store %2, %alloc[%c0, %c0, %c26] : memref<1x1x32x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    memref.store %2, %alloc[%c0, %c0, %c27] : memref<1x1x32x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    memref.store %2, %alloc[%c0, %c0, %c28] : memref<1x1x32x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    memref.store %2, %alloc[%c0, %c0, %c29] : memref<1x1x32x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    memref.store %2, %alloc[%c0, %c0, %c30] : memref<1x1x32x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    memref.store %2, %alloc[%c0, %c0, %c31] : memref<1x1x32x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
    return %alloc : memref<1x1x32x!lwe.lwe_ciphertext<encoding = #unspecified_bit_field_encoding>>
  }
}
