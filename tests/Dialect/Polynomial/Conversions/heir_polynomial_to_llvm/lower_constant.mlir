// RUN: heir-opt --polynomial-to-mod-arith %s | FileCheck %s

#ideal = #polynomial.int_polynomial<1 + x**10>
#ring = #polynomial.ring<coefficientType=!mod_arith.int<4294967296:i64>, polynomialModulus=#ideal>

func.func @test_monomial() -> !polynomial.polynomial<ring=#ring> {
  // CHECK: arith.constant dense<[1, 0, 1, 0, 0, 0, 0, 0, 0, 0]> : tensor<10xi64>
  %poly = polynomial.constant int<1 + x**2> : !polynomial.polynomial<ring=#ring>
  return %poly : !polynomial.polynomial<ring=#ring>
}
