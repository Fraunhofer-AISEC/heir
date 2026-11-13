module {
    func.func @main(%a: tensor<16xi32>, %b: tensor<16xi32>) -> i32 {
    %0 = arith.addi %a, %b : tensor<16xi32>
    return %0 : i32
    }
}