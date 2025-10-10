module {
    func.func @foo(%arg0: i8 {secret.secret}, %arg1: i8 {secret.secret}) -> i8 {
      %0 = arith.muli %arg0, %arg0 : i8
      %1 = arith.muli %0, %0 : i8
      %2 = arith.muli %1, %1 : i8
      %3 = arith.muli %2, %arg1 : i8
      return %3 : i8
    }
}