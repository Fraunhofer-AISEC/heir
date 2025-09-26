func.func @matvec_mul(%matrix: tensor<10x10xi16> {secret.secret}, %vector: tensor<10xi16> {secret.secret}) -> tensor<10xi16> {
    %result = tensor.empty() : tensor<10xi16>
    %c0_si16 = arith.constant 0 : i16
    %c10 = arith.constant 10 : index
    %final_result = affine.for %i = 0 to 10 iter_args(%res = %result) -> (tensor<10xi16>) {
        %sum = affine.for %j = 0 to 10 iter_args(%iter = %c0_si16) -> (i16) {
            %mat_elem = tensor.extract %matrix[%i, %j] : tensor<10x10xi16>
            %vec_elem = tensor.extract %vector[%j] : tensor<10xi16>
            %prod = arith.muli %mat_elem, %vec_elem : i16
            %acc = arith.addi %iter, %prod : i16
            affine.yield %acc : i16
        }
        %result_next = tensor.insert %sum into %res[%i] : tensor<10xi16>
        affine.yield %result_next : tensor<10xi16>
    }
    return %final_result : tensor<10xi16>
}