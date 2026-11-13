module{
    func.func @test(%input0: tensor<16xi8>) -> tensor<11x16xi8> {
        %R00 = tensor.extract_slice %input0 [0] [4] [1] : tensor<16xi8> to tensor<4xi8>
        %R0 = tensor.empty() : tensor<11x16xi8>
        %R1 = tensor.insert_slice %R00 into %R0 [0, 0] [1, 4] [1, 1] : tensor<4xi8> into tensor<11x16xi8>
        return %R1 : tensor<11x16xi8>
    }
}