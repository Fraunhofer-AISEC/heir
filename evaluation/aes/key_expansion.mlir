module{
    func.func @key_expansion(%Key: tensor<16xi8>) -> (tensor<11x16xi8>) {
        %rcon = arith.constant dense<[1, 2, 4, 8, 16, 32, 64, 128, 27, 54]> : tensor<10xi8>
        %c11 = arith.constant 11 : index
        %c10 = arith.constant 10 : index
        %c8 = arith.constant 8 : index
        %c4 = arith.constant 4 : index
        %c3 = arith.constant 3 : index
        %c2 = arith.constant 2 : index
        %c1 = arith.constant 1 : index
        %c0 = arith.constant 0 : index

        %R00 = tensor.extract_slice %Key [0] [4] [1] : tensor<16xi8> to tensor<4xi8>
        %R01 = tensor.extract_slice %Key [4] [4] [1] : tensor<16xi8> to tensor<4xi8>
        %R02 = tensor.extract_slice %Key [8] [4] [1] : tensor<16xi8> to tensor<4xi8>
        %R03 = tensor.extract_slice %Key [12] [4] [1] : tensor<16xi8> to tensor<4xi8>

        %R0 = tensor.empty() : tensor<11x16xi8>
        %R1 = tensor.insert_slice %R00 into %R0 [0, 0] [1, 4] [1, 1] : tensor<4xi8> into tensor<11x16xi8>
        %R2 = tensor.insert_slice %R01 into %R1 [0, 4] [1, 4] [1, 1] : tensor<4xi8> into tensor<11x16xi8>
        %R3 = tensor.insert_slice %R02 into %R2 [0, 8] [1, 4] [1, 1] : tensor<4xi8> into tensor<11x16xi8>
        %R4 = tensor.insert_slice %R03 into %R3 [0, 12] [1,4] [1, 1] : tensor<4xi8> into tensor<11x16xi8>

        %RoundKey = scf.for %i = %c0 to %c11 step %c1 iter_args(%R_iter = %R4) -> (tensor<11x16xi8>) {

            %k = arith.addi %i, %c1 : index
            %old1 = tensor.extract_slice %R_iter [%i, 0] [1, 4] [1, 1] : tensor<11x16xi8> to tensor<4xi8>
            %old2 = tensor.extract_slice %R_iter [%i, 4] [1, 4] [1, 1] : tensor<11x16xi8> to tensor<4xi8>
            %old3 = tensor.extract_slice %R_iter [%i, 8] [1, 4] [1, 1] : tensor<11x16xi8> to tensor<4xi8>
            %old4 = tensor.extract_slice %R_iter [%i, 12] [1, 4] [1, 1] : tensor<11x16xi8> to tensor<4xi8>

            %rot = func.call @rot_word(%old4) : (tensor<4xi8>) -> tensor<4xi8>
            %sub = func.call @sub_word(%rot) : (tensor<4xi8>) -> tensor<4xi8>
            %const_tensor = tensor.empty() : tensor<4xi8>
            %const_round = tensor.extract %rcon[%i] : tensor<10xi8>
            %ct1 = tensor.insert %const_round into %const_tensor[%c0] : tensor<4xi8>
            %addrcon = arith.xori %sub, %ct1 : tensor<4xi8>

            %new1 = arith.xori %addrcon, %old1 : tensor<4xi8>
            %new2 = arith.xori %new1, %old2 : tensor<4xi8>
            %new3 = arith.xori %new2, %old3 : tensor<4xi8>
            %new4 = arith.xori %new3, %old4 : tensor<4xi8>

            %newR1 = tensor.insert_slice %new1 into %R_iter[%k, 0] [1, 4] [1, 1] : tensor<4xi8> into tensor<11x16xi8>
            %newR2 = tensor.insert_slice %new2 into %newR1 [%k, 4] [1, 4] [1, 1] : tensor<4xi8> into tensor<11x16xi8>
            %newR3 = tensor.insert_slice %new3 into %newR2 [%k, 8] [1, 4] [1, 1] : tensor<4xi8> into tensor<11x16xi8>
            %newR4 = tensor.insert_slice %new4 into %newR3 [%k, 12] [1, 4] [1, 1] : tensor<4xi8> into tensor<11x16xi8>

            scf.yield %newR4 : tensor<11x16xi8>
        }  {lower = 10, upper = 10}

        return %RoundKey : tensor<11x16xi8>
    }

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

    func.func private @sub_word(%arg0: tensor<4xi8>) -> tensor<4xi8>{
        %sbox = arith.constant dense<"0x637C777BF26B6FC53001672BFED7AB76CA82C97DFA5947F0ADD4A2AF9CA472C0B7FD9326363FF7CC34A5E5F171D8311504C723C31896059A071280E2EB27B27509832C1A1B6E5AA0523BD6B329E32F8453D100ED20FCB15B6ACBBE394A4C58CFD0EFAAFB434D338545F9027F503C9FA851A3408F929D38F5BCB6DA2110FFF3D2CD0C13EC5F974417C4A77E3D645D197360814FDC222A908846EEB814DE5E0BDBE0323A0A4906245CC2D3AC629195E479E7C8376D8DD54EA96C56F4EA657AAE08BA78252E1CA6B4C6E8DD741F4BBD8B8A703EB5664803F60E613557B986C11D9EE1F8981169D98E949B1E87E9CE5528DF8CA1890DBFE6426841992D0FB054BB16"> : tensor<256xi8>

        %c5 = arith.constant 5 : index
        %c1 = arith.constant 1 : index
        %c0 = arith.constant 0 : index
        %sub = scf.for %i = %c0 to %c5 step %c1 iter_args(%iter = %arg0) -> tensor<4xi8>{
            %byte = tensor.extract %arg0[%i] : tensor<4xi8>
            %index = arith.index_cast %byte : i8 to index
            %sbox_value = tensor.extract %sbox[%index] : tensor<256xi8>
            %next = tensor.insert %sbox_value into %iter[%i] : tensor<4xi8>
            scf.yield %next : tensor<4xi8>
        } {lower = 4, upper = 4}

        return %sub : tensor<4xi8>
    }
}