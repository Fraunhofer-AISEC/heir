// WARNING: this file is autogenerated. Do not edit manually, instead see
// tests/polynomial/runner/generate_test_cases.py

//-------------------------------------------------------
// entry and check_prefix are re-set per test execution
// DEFINE: %{entry} =
// DEFINE: %{check_prefix} =

// DEFINE: %{compile} = heir-opt %s --heir-polynomial-to-llvm
// DEFINE: %{run} = mlir-runner -e %{entry} -entry-point-result=void --shared-libs="%mlir_lib_dir/libmlir_c_runner_utils%shlibext,%mlir_runner_utils"
// DEFINE: %{check} = FileCheck %s --check-prefix=%{check_prefix}
//-------------------------------------------------------

func.func private @printMemrefI32(memref<*xi32>) attributes { llvm.emit_c_interface }

// REDEFINE: %{entry} = test_6
// REDEFINE: %{check_prefix} = CHECK_TEST_6
// RUN: %{compile} | %{run} | %{check}

#ideal_6 = #polynomial.int_polynomial<3 + x**12>
!coeff_ty_6 = !mod_arith.int<16:i32>
#ring_6 = #polynomial.ring<coefficientType=!coeff_ty_6, polynomialModulus=#ideal_6>
!poly_ty_6 = !polynomial.polynomial<ring=#ring_6>

func.func @test_6() {
  %const0 = arith.constant 0 : index
  %0 = polynomial.constant int<1 + x**10> : !poly_ty_6
  %1 = polynomial.constant int<1 + x**11> : !poly_ty_6
  %2 = polynomial.mul %0, %1 : !poly_ty_6


  %3 = polynomial.to_tensor %2 : !poly_ty_6 -> tensor<12x!coeff_ty_6>
  %tensor = mod_arith.extract %3 : tensor<12x!coeff_ty_6> -> tensor<12xi32>

  %ref = bufferization.to_buffer %tensor : tensor<12xi32> to memref<12xi32>
  %U = memref.cast %ref : memref<12xi32> to memref<*xi32>
  func.call @printMemrefI32(%U) : (memref<*xi32>) -> ()
  return
}
// expected_result: Poly(x**11 + x**10 - 3*x**9 + 1, x, domain='ZZ[16]')
// CHECK_TEST_6: {{(1|-15)}}, 0, 0, 0, 0, 0, 0, 0, 0, {{(13|-3)}}, {{(1|-15)}}, {{(1|-15)}}
