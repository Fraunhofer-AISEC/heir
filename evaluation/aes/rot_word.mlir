module{
    func.func @rot_word(%arg0: tensor<4xi8>) -> tensor<4xi8> {
        %c3 = arith.constant 3 : index
        %c2 = arith.constant 2 : index
        %c1 = arith.constant 1 : index
        %c0 = arith.constant 0 : index

        %0 = tensor.extract %arg0[%c0] : tensor<4xi8>
        %1 = tensor.extract %arg0[%c1] : tensor<4xi8>
        %2 = tensor.extract %arg0[%c2] : tensor<4xi8>
        %3 = tensor.extract %arg0[%c3] : tensor<4xi8>

        %result1 = tensor.empty() : tensor<4xi8>
        %result2 = tensor.insert %1 into %result1 [%c0] : tensor<4xi8>
        %result3 = tensor.insert %2 into %result2 [%c1] : tensor<4xi8>
        %result4 = tensor.insert %3 into %result3 [%c2] : tensor<4xi8>
        %result5 = tensor.insert %0 into %result4 [%c3] : tensor<4xi8>

        return %result5 : tensor<4xi8>
    }
}