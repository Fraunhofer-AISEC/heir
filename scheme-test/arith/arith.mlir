module {
    func.func @foo(%arg0: i32 {secret.secret}, %arg1: i32 {secret.secret}) -> i32 {
      %0 = arith.addi %arg0, %arg1 : i32
      //%1 = arith.muli %arg0, %0 : i32
      return %0 : i32
    }
}