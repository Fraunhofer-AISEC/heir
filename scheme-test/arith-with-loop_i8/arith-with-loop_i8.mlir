module {
  func.func @foo(%arg0: i8 {secret.secret}, %arg1: i8 {secret.secret}) -> i8 {
    %sum = arith.addi %arg0, %arg1 : i8
    %init = arith.constant 0 : i8
    %result = affine.for %i = 0 to 3 iter_args(%acc = %init) -> (i8) {
      %prod = arith.addi %arg0, %sum : i8
      %new_acc = arith.muli %acc, %prod : i8
      affine.yield %new_acc : i8
    }
    return %result : i8
  }
}