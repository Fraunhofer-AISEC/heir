module {
    func.func @main(%input: tensor<100xi32>) -> tensor<100xi32> {
        // First affine loop: add input to itself 100 times using iter_args
        %zero_tensor = tensor.empty() : tensor<100xi32>
        %sum = affine.for %iter = 0 to 10 iter_args(%acc = %zero_tensor) -> (tensor<100xi32>) {
            %sum = affine.for %i = 0 to 10 iter_args(%inner_acc = %zero_tensor) -> (tensor<100xi32>) {
                %sum = arith.addi %inner_acc, %input : tensor<100xi32>
                affine.yield %sum : tensor<100xi32>
            }
            // Compute the product with the accumulated value
            %product = arith.muli %sum, %acc : tensor<100xi32>
            affine.yield %product : tensor<100xi32>
        }
        return %sum : tensor<100xi32>
    }
}
