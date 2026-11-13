module{
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
}