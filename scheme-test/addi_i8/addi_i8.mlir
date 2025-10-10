module {
    func.func @foo(%arg0: i8 {secret.secret}, %arg1: i8 {secret.secret}) -> i8 {
      %0 = arith.addi %arg0, %arg1 : i8
      return %0 : i8
    }
}