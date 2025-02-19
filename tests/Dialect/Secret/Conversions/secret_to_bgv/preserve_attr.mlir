// RUN: heir-opt --secret-to-bgv %s | FileCheck %s

!eui1 = !secret.secret<tensor<1024xi1>>
#mgmt = #mgmt.mgmt<level = 0, dimension = 2>
#mgmt1 = #mgmt.mgmt<level = 0, dimension = 3>

module {
  // CHECK-LABEL: func @test_preserve_attr
  func.func @test_preserve_attr(%arg0 : !eui1 {mgmt.mgmt = #mgmt}, %arg1 : !eui1 {mgmt.mgmt = #mgmt}, %arg2 : !eui1 {mgmt.mgmt = #mgmt}) -> (!eui1) {
    %0 = secret.generic ins(%arg0, %arg1 :  !eui1, !eui1) attrs = {mgmt.mgmt = #mgmt} {
    // CHECK: {dialect.attr = 1 : i64}
      ^bb0(%ARG0 : tensor<1024xi1>, %ARG1 : tensor<1024xi1>):
        %1 = arith.addi %ARG0, %ARG1 {dialect.attr = 1} : tensor<1024xi1>
        secret.yield %1 : tensor<1024xi1>
    } -> !eui1
    return %0 : !eui1
  }
}
