module {
  func.func @main(%arg0: i32 {secret.secret}, %arg1: i32 {secret.secret}) -> i32 {
    %sum = arith.addi %arg0, %arg1 : i32
    %init = arith.constant 0 : i32
    %result = affine.for %i = 0 to 3 iter_args(%acc = %init) -> (i32) {
      %prod = arith.muli %arg0, %sum : i32
      %new_acc = arith.addi %acc, %prod : i32
      affine.yield %new_acc : i32
    }
    return %result : i32
  }
}