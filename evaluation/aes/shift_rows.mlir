module{
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
}