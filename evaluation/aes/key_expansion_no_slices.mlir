module {
  func.func @key_expansion(%Key: tensor<16xi8>) -> tensor<11x16xi8> {
    %rcon = arith.constant dense<[1, 2, 4, 8, 16, 32, 64, 128, 27, 54]>
                  : tensor<10xi8>

    // Index-Konstanten
    %c0 = arith.constant 0  : index
    %c1 = arith.constant 1  : index
    %c2 = arith.constant 2  : index
    %c3 = arith.constant 3  : index
    %c4 = arith.constant 4  : index
    %c8 = arith.constant 8  : index
    %c10 = arith.constant 10 : index
    %c11 = arith.constant 11 : index
    %c12 = arith.constant 12 : index
    %c16 = arith.constant 16 : index

    %R0 = tensor.empty() : tensor<11x16xi8>
    %R1 = func.call @insert_key(%R0, %Key) : (tensor<11x16xi8>, tensor<16xi8>) -> tensor<11x16xi8>

    // -------------------------------------------------------------------
    // 2. Schlüssel­expansions­schleife
    // -------------------------------------------------------------------
    %RoundKey = scf.for %i = %c0 to %c10 step %c1 iter_args(%R_iter = %R1) -> tensor<11x16xi8> {

      %k = arith.addi %i, %c1 : index   // Zeile, die wir gerade erzeugen

      %old1 = func.call @extract_word(%R_iter, %i, %c0) : (tensor<11x16xi8>, index, index) -> tensor<4xi8>
      %old2 = func.call @extract_word(%R_iter, %i, %c4) : (tensor<11x16xi8>, index, index) -> tensor<4xi8>
      %old3 = func.call @extract_word(%R_iter, %i, %c8) : (tensor<11x16xi8>, index, index) -> tensor<4xi8>
      %old4 = func.call @extract_word(%R_iter, %i, %c12) : (tensor<11x16xi8>, index, index) -> tensor<4xi8>

      // ------------------------------------------------------------
      // 2.2 AES-spezifische Operationen
      // ------------------------------------------------------------
      %rot  = func.call @rot_word(%old4) : (tensor<4xi8>) -> tensor<4xi8>
      %sub  = func.call @sub_word(%rot)  : (tensor<4xi8>) -> tensor<4xi8>

      // RCON einfügen
      %const_tensor = tensor.empty() : tensor<4xi8>
      %const_round = tensor.extract %rcon[%i] : tensor<10xi8>
      %ct1 = tensor.insert %const_round into %const_tensor[%c0] : tensor<4xi8>
      %addrcon = arith.xori %sub, %ct1 : tensor<4xi8>

      %new1 = arith.xori %addrcon, %old1 : tensor<4xi8>
      %new2 = arith.xori %new1,  %old2  : tensor<4xi8>
      %new3 = arith.xori %new2,  %old3  : tensor<4xi8>
      %new4 = arith.xori %new3,  %old4  : tensor<4xi8>
    
      %tmp1 = func.call @insert_word(%R_iter, %new1, %k, %c0) : (tensor<11x16xi8>, tensor<4xi8>, index, index) -> tensor<11x16xi8>
      %tmp2 = func.call @insert_word(%tmp1, %new2, %k, %c4) : (tensor<11x16xi8>, tensor<4xi8>, index, index) -> tensor<11x16xi8>
      %tmp3 = func.call @insert_word(%tmp2, %new3, %k, %c8) : (tensor<11x16xi8>, tensor<4xi8>, index, index) -> tensor<11x16xi8>
      %newR4 = func.call @insert_word(%tmp3, %new4, %k, %c12) : (tensor<11x16xi8>, tensor<4xi8>, index, index) -> tensor<11x16xi8>

      scf.yield %newR4 : tensor<11x16xi8>
    }

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

    func.func private @extract_word(%arg0: tensor<11x16xi8>, %row: index, %offset: index) -> tensor<4xi8>{
        %c0 = arith.constant 0  : index
        %c1 = arith.constant 1  : index
        %c2 = arith.constant 2  : index
        %c3 = arith.constant 3  : index

        %tmp = tensor.empty() : tensor<4xi8>

        %b0 = tensor.extract %arg0[%row, %offset]          : tensor<11x16xi8>
        %t1 = tensor.insert %b0 into %tmp[%c0]              : tensor<4xi8>

        %b1c = arith.addi %offset, %c1 : index
        %b1  = tensor.extract %arg0[%row, %b1c]             : tensor<11x16xi8>
        %t2  = tensor.insert %b1 into %t1[%c1]              : tensor<4xi8>

        %b2c = arith.addi %offset, %c2 : index
        %b2  = tensor.extract %arg0[%row, %b2c]             : tensor<11x16xi8>
        %t3  = tensor.insert %b2 into %t2[%c2]              : tensor<4xi8>

        %b3c = arith.addi %offset, %c3 : index
        %b3  = tensor.extract %arg0[%row, %b3c]             : tensor<11x16xi8>
        %t4  = tensor.insert %b3 into %t3[%c3]              : tensor<4xi8>

        return %t4 : tensor<4xi8>      
    }

    func.func private @insert_key(%arg0: tensor<11x16xi8>, %arg1: tensor<16xi8>) -> tensor<11x16xi8>{
        %c0 = arith.constant 0  : index
        %c1 = arith.constant 1  : index
        %c2 = arith.constant 2  : index
        %c3 = arith.constant 3  : index
        %c4 = arith.constant 4  : index
        %c5 = arith.constant 5  : index
        %c6 = arith.constant 6  : index
        %c7 = arith.constant 7  : index
        %c8 = arith.constant 8  : index
        %c9 = arith.constant 9  : index
        %c10 = arith.constant 10 : index
        %c11 = arith.constant 11 : index
        %c12 = arith.constant 12 : index
        %c13 = arith.constant 13 : index
        %c14 = arith.constant 14 : index
        %c15 = arith.constant 15 : index
        
        %b0 = tensor.extract %arg1[%c0]                      : tensor<16xi8>
        %t1 = tensor.insert %b0 into %arg0[%c0, %c0]        : tensor<11x16xi8>

        %b1  = tensor.extract %arg1[%c1]                     : tensor<16xi8>
        %t2  = tensor.insert %b1 into %t1[%c0, %c1]         : tensor<11x16xi8>

        %b2  = tensor.extract %arg1[%c2]                     : tensor<16xi8>
        %t3  = tensor.insert %b2 into %t2[%c0, %c2]         : tensor<11x16xi8>

        %b3  = tensor.extract %arg1[%c3]                    : tensor<16xi8>
        %t4  = tensor.insert %b3 into %t3[%c0, %c3]         : tensor<11x16xi8>

        %b4  = tensor.extract %arg1[%c4]                    : tensor<16xi8>
        %t5  = tensor.insert %b4 into %t4[%c0, %c3]         : tensor<11x16xi8>

        %b5  = tensor.extract %arg1[%c5]                    : tensor<16xi8>
        %t6  = tensor.insert %b5 into %t5[%c0, %c3]         : tensor<11x16xi8>

        %b6  = tensor.extract %arg1[%c6]                    : tensor<16xi8>
        %t7  = tensor.insert %b6 into %t6[%c0, %c3]         : tensor<11x16xi8>

        %b7  = tensor.extract %arg1[%c7]                    : tensor<16xi8>
        %t8  = tensor.insert %b7 into %t7[%c0, %c3]         : tensor<11x16xi8>

        %b8  = tensor.extract %arg1[%c8]                    : tensor<16xi8>
        %t9  = tensor.insert %b8 into %t8[%c0, %c3]         : tensor<11x16xi8>

        %b9  = tensor.extract %arg1[%c9]                    : tensor<16xi8>
        %t10  = tensor.insert %b9 into %t9[%c0, %c3]         : tensor<11x16xi8>

        %b10  = tensor.extract %arg1[%c10]                    : tensor<16xi8>
        %t11  = tensor.insert %b10 into %t10[%c0, %c3]         : tensor<11x16xi8>

        %b11  = tensor.extract %arg1[%c11]                    : tensor<16xi8>
        %t12  = tensor.insert %b11 into %t11[%c0, %c3]         : tensor<11x16xi8>

        %b12  = tensor.extract %arg1[%c12]                    : tensor<16xi8>
        %t13  = tensor.insert %b12 into %t12[%c0, %c3]         : tensor<11x16xi8>

        %b13  = tensor.extract %arg1[%c13]                    : tensor<16xi8>
        %t14  = tensor.insert %b13 into %t13[%c0, %c3]         : tensor<11x16xi8>

        %b14  = tensor.extract %arg1[%c14]                    : tensor<16xi8>
        %t15  = tensor.insert %b14 into %t14[%c0, %c3]         : tensor<11x16xi8>

        return %t15 : tensor<11x16xi8>
    }

    func.func private @insert_word(%arg0: tensor<11x16xi8>, %arg1: tensor<4xi8>, %row: index, %offset: index) -> tensor<11x16xi8>{
        %c0 = arith.constant 0  : index
        %c1 = arith.constant 1  : index
        %c2 = arith.constant 2  : index
        %c3 = arith.constant 3  : index

        %b0 = tensor.extract %arg1[%c0]                     : tensor<4xi8>
        %t1 = tensor.insert %b0 into %arg0[%row, %offset]    : tensor<11x16xi8>

        %b1c = arith.addi %offset, %c1 : index
        %b1  = tensor.extract %arg1[%c1]                    : tensor<4xi8>
        %t2  = tensor.insert %b1 into %t1[%row, %b1c]         : tensor<11x16xi8>

        %b2c = arith.addi %offset, %c2 : index
        %b2  = tensor.extract %arg1[%c2]                    : tensor<4xi8>
        %t3  = tensor.insert %b2 into %t2[%row, %b2c]         : tensor<11x16xi8>

        %b3c = arith.addi %offset, %c3 : index
        %b3  = tensor.extract %arg1[%c3]                    : tensor<4xi8>
        %t4  = tensor.insert %b3 into %t3[%row, %b3c]         : tensor<11x16xi8>

        return %t4 : tensor<11x16xi8>
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