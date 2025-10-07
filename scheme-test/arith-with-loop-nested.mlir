module {
  func.func @main(%arg0: i32, %arg1: i32) -> i32 {
    %sum = arith.addi %arg0, %arg1 : i32
    %init = arith.constant 0 : i32
    %result = affine.for %i = 0 to 1 iter_args(%acc_outer = %init) -> (i32) {
      %inner_sum = affine.for %j = 0 to 1 iter_args(%acc_inner = %acc_outer) -> (i32) {
        %prod = arith.muli %arg0, %sum : i32
        %new_acc = arith.addi %acc_inner, %prod : i32
        affine.yield %new_acc : i32
      }
      %outer_sum = arith.addi %sum, %acc_outer : i32
      affine.yield %outer_sum : i32
    }
    return %result : i32
  }
}