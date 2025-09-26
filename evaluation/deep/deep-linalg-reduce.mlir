module {
    func.func @main(%input: tensor<4x3xi32>) -> tensor<3xi32> {
        %init_tensor = tensor.empty() : tensor<3xi32>
        %result = linalg.reduce
            ins(%input : tensor<4x3xi32>)
            outs(%init_tensor : tensor<3xi32>)
            dimensions = [0]
            (%in: i32, %out: i32) {
                %sum = arith.addi %out, %in : i32
                linalg.yield %sum : i32
            }
        return %result : tensor<3xi32>
    }
}
