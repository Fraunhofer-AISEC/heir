module {
    func.func @ctr_mode(%key: tensor<16xi8>, %iv: tensor<16xi8>, %data: tensor<?x16xi8>) -> (tensor<?x16xi8>) {
        %c255 = arith.constant 255: i8
        %c16 = arith.constant 16 : index
        %c15 = arith.constant 15 : index
        %c1 = arith.constant 1 : index
        %c0 = arith.constant 0 : index
        %c0_i8 = arith.constant 0 : i8
        %c1_i8 = arith.constant 1 : i8

        %true = arith.constant 1 : i1
        %false = arith.constant 0 : i1

        %data_len = tensor.dim %data, %c0 : tensor<?x16xi8>

        %round_keys = func.call @key_expansion(%key) : (tensor<16xi8>) -> tensor<11x16xi8>

        %enc_loop_data, %looped_iv = scf.for %i = %c0 to %data_len step %c1 iter_args(%enc_data = %data, %cur_iv = %iv) -> (tensor<?x16xi8>, tensor<16xi8>) {

            %encrypted_block = func.call @encrypt_block(%cur_iv, %round_keys) : (tensor<16xi8>, tensor<11x16xi8>) -> tensor<16xi8>
            %data_block = tensor.extract_slice %data [%i, 0] [1, 16] [1, 1] : tensor<?x16xi8> to tensor<16xi8>

            %data_iter = tensor.insert_slice %encrypted_block into %enc_data [%i, 0] [1, 16] [1, 1] : tensor<16xi8> into tensor<?x16xi8>
            %next_iv, %loop_overflow = scf.for %j = %c0 to %c16 step %c1 iter_args(%inc_iter_iv = %cur_iv, %prev_overflow = %true) -> (tensor<16xi8>, i1) {

                %k = arith.subi %c15, %j : index
                // Load the current byte of the IV
                %iv_byte = tensor.extract %inc_iter_iv[%k] : tensor<16xi8>

                // Check if the current byte is equal to 255

                %is_overflow = arith.cmpi eq, %iv_byte, %c255 : i8

                %inc_iv, %old_overflow = scf.if %prev_overflow -> (tensor<16xi8>, i1){
                    %overflow_iv, %overflow = scf.if %is_overflow -> (tensor<16xi8>, i1) {
                        // Set the current byte to 0
                        %zero_byte = arith.constant 0 : i8
                        %incremented_iv = tensor.insert %zero_byte into %inc_iter_iv[%k] : tensor<16xi8>
                        scf.yield %incremented_iv, %true : tensor<16xi8>, i1
                    } else {
                        %incremented_byte = arith.addi %iv_byte, %c1_i8 : i8
                        %incremented_iv = tensor.insert %incremented_byte into %inc_iter_iv[%k] : tensor<16xi8>
                        scf.yield %incremented_iv, %false : tensor<16xi8>, i1
                    }
                    scf.yield %overflow_iv, %overflow : tensor<16xi8>, i1
                } else {
                    scf.yield %inc_iter_iv, %false: tensor<16xi8>, i1
                }

                scf.yield %inc_iv, %old_overflow : tensor<16xi8>, i1
            }  {lower = 16, upper = 16}
            scf.yield %data_iter, %next_iv : tensor<?x16xi8>, tensor<16xi8>
        } {lower = 0, upper = 64}
        // Return the result
        return %enc_loop_data : tensor<?x16xi8>
    }

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

            // Füge die Rotword-Ausgaben in einen neuen RoundKeys-Tensor ein
            %newR1 = tensor.insert_slice %new1 into %R_iter[%k, 0] [1, 4] [1, 1] : tensor<4xi8> into tensor<11x16xi8>
            %newR2 = tensor.insert_slice %new2 into %newR1 [%k, 4] [1, 4] [1, 1] : tensor<4xi8> into tensor<11x16xi8>
            %newR3 = tensor.insert_slice %new3 into %newR2 [%k, 8] [1, 4] [1, 1] : tensor<4xi8> into tensor<11x16xi8>
            %newR4 = tensor.insert_slice %new4 into %newR3 [%k, 12] [1, 4] [1, 1] : tensor<4xi8> into tensor<11x16xi8>

            scf.yield %newR4 : tensor<11x16xi8>
        }  {lower = 10, upper = 10}

        return %RoundKey : tensor<11x16xi8>
    }

    func.func @encrypt_block(%state: tensor<16xi8>, %round_keys: tensor<11x16xi8>) -> (tensor<16xi8>) {
        %c1 = arith.constant 1 : index
        %c9 = arith.constant 9 : index
        %c10 = arith.constant 10 : index
        %c11 = arith.constant 11 : index

        %firstKey = tensor.extract_slice %round_keys [0, 0] [1, 16] [1, 1] : tensor<11x16xi8> to tensor<16xi8>
        %addfirstkey = arith.xori %state, %firstKey : tensor<16xi8>
        %stateblock = func.call @blockstate(%addfirstkey) : (tensor<16xi8>) -> tensor<4x4xi8>


        %modstate = scf.for %i = %c1 to %c10 step %c1 iter_args(%iter = %stateblock) -> (tensor<4x4xi8>){

            %sub_bytes = func.call @sub_bytes(%iter) : (tensor<4x4xi8>) -> tensor<4x4xi8>
            %shifted_state = func.call @shift_rows(%sub_bytes) : (tensor<4x4xi8>) -> tensor<4x4xi8>
            %mixed = func.call @mix_columns(%shifted_state) : (tensor<4x4xi8>) -> tensor<4x4xi8>

            %roundkey = tensor.extract_slice %round_keys [%i, 0] [1, 16] [1, 1] : tensor<11x16xi8> to tensor<16xi8>
            %roundkeyblock = func.call @blockstate(%roundkey) : (tensor<16xi8>) -> tensor<4x4xi8>
            %addroundkey = arith.xori %mixed, %roundkeyblock : tensor<4x4xi8>

            scf.yield %addroundkey : tensor<4x4xi8>
        }  {lower = 9, upper = 9}

        %final_sub_bytes = func.call @sub_bytes(%modstate) : (tensor<4x4xi8>) -> tensor<4x4xi8>
        %final_shift = func.call @shift_rows(%final_sub_bytes) : (tensor<4x4xi8>) -> tensor<4x4xi8>
        %final_roundkey = tensor.extract_slice %round_keys [%c10, 0] [1, 16] [1, 1] : tensor<11x16xi8> to tensor<16xi8>
        %final_roundkeyblock = func.call @blockstate(%final_roundkey) : (tensor<16xi8>) -> tensor<4x4xi8>
        %final_add = arith.xori %final_shift, %final_roundkeyblock : tensor<4x4xi8>
        %flattened = func.call @flattenstate(%final_add) : (tensor<4x4xi8>) -> tensor<16xi8>
        return %flattened : tensor<16xi8>
    }

    func.func private @flattenstate(%arg: tensor<4x4xi8>) -> (tensor<16xi8>){
        %empty = tensor.empty() : tensor<16xi8>
        %row0 = tensor.extract_slice %arg [0, 0] [4, 1] [1, 1] : tensor<4x4xi8> to tensor<4xi8>
        %row1 = tensor.extract_slice %arg [0, 1] [4, 1] [1, 1] : tensor<4x4xi8> to tensor<4xi8>
        %row2 = tensor.extract_slice %arg [0, 2] [4, 1] [1, 1] : tensor<4x4xi8> to tensor<4xi8>
        %row3 = tensor.extract_slice %arg [0, 3] [4, 1] [1, 1] : tensor<4x4xi8> to tensor<4xi8>

        %block1 = tensor.insert_slice %row0 into %empty  [0] [4] [1] : tensor<4xi8> into tensor<16xi8>
        %block2 = tensor.insert_slice %row1 into %block1 [4] [4] [1] : tensor<4xi8> into tensor<16xi8>
        %block3 = tensor.insert_slice %row2 into %block2 [8] [4] [1] : tensor<4xi8> into tensor<16xi8>
        %block4 = tensor.insert_slice %row3 into %block3 [12] [4] [1] : tensor<4xi8> into tensor<16xi8>

        return %block4 : tensor<16xi8>
    }

    func.func private @blockstate(%arg: tensor<16xi8>) -> (tensor<4x4xi8>){
        %empty = tensor.empty() : tensor<4x4xi8>
        %row0 = tensor.extract_slice %arg [0] [4] [1] : tensor<16xi8> to tensor<4xi8>
        %row1 = tensor.extract_slice %arg [4] [4] [1] : tensor<16xi8> to tensor<4xi8>
        %row2 = tensor.extract_slice %arg [8] [4] [1] : tensor<16xi8> to tensor<4xi8>
        %row3 = tensor.extract_slice %arg [12][4] [1] : tensor<16xi8> to tensor<4xi8>

        %block1 = tensor.insert_slice %row0 into %empty  [0, 0] [4, 1] [1, 1] : tensor<4xi8> into tensor<4x4xi8>
        %block2 = tensor.insert_slice %row1 into %block1 [0, 1] [4, 1] [1, 1] : tensor<4xi8> into tensor<4x4xi8>
        %block3 = tensor.insert_slice %row2 into %block2 [0, 2] [4, 1] [1, 1] : tensor<4xi8> into tensor<4x4xi8>
        %block4 = tensor.insert_slice %row3 into %block3 [0, 3] [4, 1] [1, 1] : tensor<4xi8> into tensor<4x4xi8>

        return %block4 : tensor<4x4xi8>
    }

    func.func @sub_bytes(%state: tensor<4x4xi8>) -> tensor<4x4xi8> {
        %sbox = arith.constant dense<"0x637C777BF26B6FC53001672BFED7AB76CA82C97DFA5947F0ADD4A2AF9CA472C0B7FD9326363FF7CC34A5E5F171D8311504C723C31896059A071280E2EB27B27509832C1A1B6E5AA0523BD6B329E32F8453D100ED20FCB15B6ACBBE394A4C58CFD0EFAAFB434D338545F9027F503C9FA851A3408F929D38F5BCB6DA2110FFF3D2CD0C13EC5F974417C4A77E3D645D197360814FDC222A908846EEB814DE5E0BDBE0323A0A4906245CC2D3AC629195E479E7C8376D8DD54EA96C56F4EA657AAE08BA78252E1CA6B4C6E8DD741F4BBD8B8A703EB5664803F60E613557B986C11D9EE1F8981169D98E949B1E87E9CE5528DF8CA1890DBFE6426841992D0FB054BB16"> : tensor<256xi8>
        %c5 = arith.constant 5 : index
        %c1 = arith.constant 1 : index
        %c0 = arith.constant 0 : index

        %outer = scf.for %i = %c0 to %c5 step %c1 iter_args(%iter1 = %state) -> (tensor<4x4xi8>){
          %inner = scf.for %j = %c0 to %c5 step %c1 iter_args(%iter2 = %iter1) -> (tensor<4x4xi8>){
            %value = tensor.extract %iter2[%j, %i] : tensor<4x4xi8>
            %index = arith.index_cast %value : i8 to index
            %sbox_value = tensor.extract %sbox[%index] : tensor<256xi8>
            %next = tensor.insert %sbox_value into %iter2[%j, %i] : tensor<4x4xi8>
            scf.yield %next : tensor<4x4xi8>
          }   {lower = 4, upper = 4}
          scf.yield %inner : tensor<4x4xi8>
        }   {lower = 4, upper = 4}

        return %outer: tensor<4x4xi8>
    }

    func.func @shift_rows(%state: tensor<4x4xi8>) -> (tensor<4x4xi8>) {
        %c3 = arith.constant 3 : index
        %c2 = arith.constant 2 : index
        %c1 = arith.constant 1 : index
        %c0 = arith.constant 0 : index

        // Create new tensors to hold the shifted rows
        %new_row0 = tensor.extract_slice %state [0, 0] [4, 1] [1, 1] : tensor<4x4xi8> to tensor<4xi8>
        %new_row1 = tensor.extract_slice %state [0, 1] [4, 1] [1, 1] : tensor<4x4xi8> to tensor<4xi8>
        %new_row2 = tensor.extract_slice %state [0, 2] [4, 1] [1, 1] : tensor<4x4xi8> to tensor<4xi8>
        %new_row3 = tensor.extract_slice %state [0, 3] [4, 1] [1, 1] : tensor<4x4xi8> to tensor<4xi8>

        // Row 0: No shift
        %value0_0 = tensor.extract %new_row0[%c0] : tensor<4xi8>
        %value0_1 = tensor.extract %new_row0[%c1] : tensor<4xi8>
        %value0_2 = tensor.extract %new_row0[%c2] : tensor<4xi8>
        %value0_3 = tensor.extract %new_row0[%c3] : tensor<4xi8>
        %shifted00 = tensor.insert %value0_0 into %new_row0[%c0] : tensor<4xi8>
        %shifted01 = tensor.insert %value0_1 into %shifted00[%c1] : tensor<4xi8>
        %shifted02 = tensor.insert %value0_2 into %shifted01[%c2] : tensor<4xi8>
        %shifted03 = tensor.insert %value0_3 into %shifted02[%c3] : tensor<4xi8>

        // Row 1: Shift left by 1
        %value1_0 = tensor.extract %new_row1[%c0] : tensor<4xi8>
        %value1_1 = tensor.extract %new_row1[%c1] : tensor<4xi8>
        %value1_2 = tensor.extract %new_row1[%c2] : tensor<4xi8>
        %value1_3 = tensor.extract %new_row1[%c3] : tensor<4xi8>
        %shifted10 = tensor.insert %value1_1 into %new_row1[%c0] : tensor<4xi8>
        %shifted11 = tensor.insert %value1_2 into %shifted10[%c1] : tensor<4xi8>
        %shifted12 = tensor.insert %value1_3 into %shifted11[%c2] : tensor<4xi8>
        %shifted13 = tensor.insert %value1_0 into %shifted12[%c3] : tensor<4xi8>

        // Row 2: Shift left by 2
        %value2_0 = tensor.extract %new_row2[%c0] : tensor<4xi8>
        %value2_1 = tensor.extract %new_row2[%c1] : tensor<4xi8>
        %value2_2 = tensor.extract %new_row2[%c2] : tensor<4xi8>
        %value2_3 = tensor.extract %new_row2[%c3] : tensor<4xi8>
        %shifted20 = tensor.insert %value2_2 into %new_row2[%c0] : tensor<4xi8>
        %shifted21 = tensor.insert %value2_3 into %shifted20[%c1] : tensor<4xi8>
        %shifted22 = tensor.insert %value2_0 into %shifted21[%c2] : tensor<4xi8>
        %shifted23 = tensor.insert %value2_1 into %shifted22[%c3] : tensor<4xi8>

        // Row 3: Shift left by 3
        %value3_0 = tensor.extract %new_row3[%c0] : tensor<4xi8>
        %value3_1 = tensor.extract %new_row3[%c1] : tensor<4xi8>
        %value3_2 = tensor.extract %new_row3[%c2] : tensor<4xi8>
        %value3_3 = tensor.extract %new_row3[%c3] : tensor<4xi8>
        %shifted30 = tensor.insert %value3_3 into %new_row3[%c0] : tensor<4xi8>
        %shifted31 = tensor.insert %value3_0 into %shifted30[%c1] : tensor<4xi8>
        %shifted32 = tensor.insert %value3_1 into %shifted31[%c2] : tensor<4xi8>
        %shifted33 = tensor.insert %value3_2 into %shifted32[%c3] : tensor<4xi8>

        %new_state1 = tensor.insert_slice %shifted03 into %state [0, 0] [4, 1] [1, 1] : tensor<4xi8> into tensor<4x4xi8>
        %new_state2 = tensor.insert_slice %shifted13 into %state [0, 1] [4, 1] [1, 1] : tensor<4xi8> into tensor<4x4xi8>
        %new_state3 = tensor.insert_slice %shifted23 into %state [0, 2] [4, 1] [1, 1] : tensor<4xi8> into tensor<4x4xi8>
        %new_state4 = tensor.insert_slice %shifted33 into %state [0, 3] [4, 1] [1, 1] : tensor<4xi8> into tensor<4x4xi8>


        // Return the new shifted rows
        return %new_state4: tensor<4x4xi8>
    }

    func.func @mix_columns(%arg: tensor<4x4xi8>) -> (tensor<4x4xi8>){
        %c4 = arith.constant 4 : index
        %c3 = arith.constant 3 : index
        %c2 = arith.constant 2 : index
        %c1 = arith.constant 1 : index
        %c0 = arith.constant 0 : index
        %c0_i8 = arith.constant 0 : i8

        %empty = tensor.empty() : tensor<4x4xi8>

        %col0 = tensor.extract_slice %arg [0, 0] [1, 4] [1, 1] : tensor<4x4xi8> to tensor<4xi8>
        %col1 = tensor.extract_slice %arg [1, 0] [1, 4] [1, 1] : tensor<4x4xi8> to tensor<4xi8>
        %col2 = tensor.extract_slice %arg [2, 0] [1, 4] [1, 1] : tensor<4x4xi8> to tensor<4xi8>
        %col3 = tensor.extract_slice %arg [3, 0] [1, 4] [1, 1] : tensor<4x4xi8> to tensor<4xi8>

        %temp1 = arith.xori %col0, %col1 : tensor<4xi8>
        %temp2 = arith.xori %col2, %col3 : tensor<4xi8>
        %temp3 = arith.xori %temp1, %temp2 : tensor<4xi8>

        %mixed = scf.for %i = %c0 to %c4 step %c1 iter_args(%iter = %arg) -> (tensor<4x4xi8>){

            %tmp = tensor.extract %temp3[%i] : tensor<4xi8>

            %item0 = tensor.extract %iter[%i, %c0] : tensor<4x4xi8>
            %item1 = tensor.extract %iter[%i, %c1] : tensor<4x4xi8>
            %tm10 = arith.xori %item0, %item1 : i8
            %xtime1 = func.call @xtime(%tm10) : (i8) -> i8
            %tm10tmp = arith.xori %xtime1, %tmp : i8
            %mod1 = arith.xori %item0, %tm10tmp : i8
            %next1 = tensor.insert %mod1 into %iter[%i, %c0] : tensor<4x4xi8>

            %item2 = tensor.extract %iter[%i, %c2] : tensor<4x4xi8>
            %tm12 = arith.xori %item2, %item1 : i8
            %xtime2 = func.call @xtime(%tm12) : (i8) -> i8
            %tm12tmp = arith.xori %xtime2, %tmp : i8
            %mod2 = arith.xori %item1, %tm12tmp : i8
            %next2 = tensor.insert %mod2 into %iter[%i, %c1] : tensor<4x4xi8>

            %item3 = tensor.extract %iter[%i, %c3] : tensor<4x4xi8>
            %tm32 = arith.xori %item2, %item3 : i8
            %xtime3 = func.call @xtime(%tm32) : (i8) -> i8
            %tm32tmp = arith.xori %xtime3, %tmp : i8
            %mod3 = arith.xori %item2, %tm32tmp : i8
            %next3 = tensor.insert %mod3 into %iter[%i, %c2] : tensor<4x4xi8>

            %tm30 = arith.xori %item0, %item3 : i8
            %xtime4 = func.call @xtime(%tm30) : (i8) -> i8
            %tm30tmp = arith.xori %xtime4, %tmp : i8
            %mod4 = arith.xori %item3, %tm30tmp : i8
            %next4 = tensor.insert %mod4 into %iter[%i, %c3] : tensor<4x4xi8>

            scf.yield %next4 : tensor<4x4xi8>
        }  {lower = 4, upper = 4}
        return %mixed : tensor<4x4xi8>
    }

    func.func private @xtime(%arg0: i8) -> i8 {
        %c1 = arith.constant 1 : i8
        %c7 = arith.constant 7: i8
        %c1b = arith.constant 27 : i8  // 0x1B in decimal
        %shifted = arith.shli %arg0, %c1 : i8   // Shift left by 1
        %msb = arith.shrsi %arg0, %c7 : i8        // Right shift by 7 to isolate the MSB
        %msb_check = arith.andi %msb, %c1 : i8   // AND with 1 to get 0 or 1

        // Multiply the MSB check by 0x1B (which is 27)
        %result = arith.muli %msb_check, %c1b : i8

        // XOR the shifted value with the result of the multiplication
        %final_result = arith.xori %shifted, %result : i8

        return %final_result : i8
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