module {
    func.func @foo(%arg0: i32 {secret.secret}, %arg1: i32 {secret.secret}) -> i32 {
      %0 = arith.muli %arg0, %arg0 : i32
      %1 = arith.muli %0, %0 : i32
      %2 = arith.muli %1, %1 : i32
      %3 = arith.muli %2, %arg1 : i32
      return %3 : i32
    }
}