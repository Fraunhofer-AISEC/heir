module {
  func.func @key_expansion(%arg0: tensor<16xi8> {secret.secret}) -> tensor<11x16xi8> {
    %c5 = arith.constant 5 : index
    %cst = arith.constant dense<"0x637C777BF26B6FC53001672BFED7AB76CA82C97DFA5947F0ADD4A2AF9CA472C0B7FD9326363FF7CC34A5E5F171D8311504C723C31896059A071280E2EB27B27509832C1A1B6E5AA0523BD6B329E32F8453D100ED20FCB15B6ACBBE394A4C58CFD0EFAAFB434D338545F9027F503C9FA851A3408F929D38F5BCB6DA2110FFF3D2CD0C13EC5F974417C4A77E3D645D197360814FDC222A908846EEB814DE5E0BDBE0323A0A4906245CC2D3AC629195E479E7C8376D8DD54EA96C56F4EA657AAE08BA78252E1CA6B4C6E8DD741F4BBD8B8A703EB5664803F60E613557B986C11D9EE1F8981169D98E949B1E87E9CE5528DF8CA1890DBFE6426841992D0FB054BB16"> : tensor<256xi8>
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %cst_0 = arith.constant dense<[1, 2, 4, 8, 16, 32, 64, -128, 27, 54]> : tensor<10xi8>
    %c11 = arith.constant 11 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %extracted_slice = tensor.extract_slice %arg0[0] [4] [1] : tensor<16xi8> to tensor<4xi8>
    %extracted_slice_1 = tensor.extract_slice %arg0[4] [4] [1] : tensor<16xi8> to tensor<4xi8>
    %extracted_slice_2 = tensor.extract_slice %arg0[8] [4] [1] : tensor<16xi8> to tensor<4xi8>
    %extracted_slice_3 = tensor.extract_slice %arg0[12] [4] [1] : tensor<16xi8> to tensor<4xi8>
    %0 = tensor.empty() : tensor<11x16xi8>
    %inserted_slice = tensor.insert_slice %extracted_slice into %0[0, 0] [1, 4] [1, 1] : tensor<4xi8> into tensor<11x16xi8>
    %inserted_slice_4 = tensor.insert_slice %extracted_slice_1 into %inserted_slice[0, 4] [1, 4] [1, 1] : tensor<4xi8> into tensor<11x16xi8>
    %inserted_slice_5 = tensor.insert_slice %extracted_slice_2 into %inserted_slice_4[0, 8] [1, 4] [1, 1] : tensor<4xi8> into tensor<11x16xi8>
    %inserted_slice_6 = tensor.insert_slice %extracted_slice_3 into %inserted_slice_5[0, 12] [1, 4] [1, 1] : tensor<4xi8> into tensor<11x16xi8>
    %1 = scf.for %arg1 = %c0 to %c11 step %c1 iter_args(%arg2 = %inserted_slice_6) -> (tensor<11x16xi8>) {
      %2 = arith.addi %arg1, %c1 : index
      %extracted_slice_7 = tensor.extract_slice %arg2[%arg1, 0] [1, 4] [1, 1] : tensor<11x16xi8> to tensor<4xi8>
      %extracted_slice_8 = tensor.extract_slice %arg2[%arg1, 4] [1, 4] [1, 1] : tensor<11x16xi8> to tensor<4xi8>
      %extracted_slice_9 = tensor.extract_slice %arg2[%arg1, 8] [1, 4] [1, 1] : tensor<11x16xi8> to tensor<4xi8>
      %extracted_slice_10 = tensor.extract_slice %arg2[%arg1, 12] [1, 4] [1, 1] : tensor<11x16xi8> to tensor<4xi8>
      %extracted = tensor.extract %extracted_slice_10[%c0] : tensor<4xi8>
      %extracted_11 = tensor.extract %extracted_slice_10[%c1] : tensor<4xi8>
      %extracted_12 = tensor.extract %extracted_slice_10[%c2] : tensor<4xi8>
      %extracted_13 = tensor.extract %extracted_slice_10[%c3] : tensor<4xi8>
      %from_elements = tensor.from_elements %extracted_11, %extracted_12, %extracted_13, %extracted : tensor<4xi8>
      %3:4 = scf.for %arg3 = %c0 to %c5 step %c1 iter_args(%arg4 = %extracted_11, %arg5 = %extracted_12, %arg6 = %extracted_13, %arg7 = %extracted) -> (i8, i8, i8, i8) {
        %from_elements_24 = tensor.from_elements %arg4, %arg5, %arg6, %arg7 : tensor<4xi8>
        %extracted_25 = tensor.extract %from_elements[%arg3] : tensor<4xi8>
        %10 = arith.index_cast %extracted_25 : i8 to index
        %extracted_26 = tensor.extract %cst[%10] : tensor<256xi8>
        %inserted = tensor.insert %extracted_26 into %from_elements_24[%arg3] : tensor<4xi8>
        %extracted_27 = tensor.extract %inserted[%c0] : tensor<4xi8>
        %extracted_28 = tensor.extract %inserted[%c1] : tensor<4xi8>
        %extracted_29 = tensor.extract %inserted[%c2] : tensor<4xi8>
        %extracted_30 = tensor.extract %inserted[%c3] : tensor<4xi8>
        scf.yield %extracted_27, %extracted_28, %extracted_29, %extracted_30 : i8, i8, i8, i8
      } {lower = 4 : i64, upper = 4 : i64}
      %from_elements_14 = tensor.from_elements %3#0, %3#1, %3#2, %3#3 : tensor<4xi8>
      %4 = tensor.empty() : tensor<4xi8>
      %extracted_15 = tensor.extract %4[%c1] : tensor<4xi8>
      %extracted_16 = tensor.extract %4[%c2] : tensor<4xi8>
      %extracted_17 = tensor.extract %4[%c3] : tensor<4xi8>
      %extracted_18 = tensor.extract %cst_0[%arg1] : tensor<10xi8>
      %from_elements_19 = tensor.from_elements %extracted_18, %extracted_15, %extracted_16, %extracted_17 : tensor<4xi8>
      %5 = arith.xori %from_elements_14, %from_elements_19 : tensor<4xi8>
      %6 = arith.xori %5, %extracted_slice_7 : tensor<4xi8>
      %7 = arith.xori %6, %extracted_slice_8 : tensor<4xi8>
      %8 = arith.xori %7, %extracted_slice_9 : tensor<4xi8>
      %9 = arith.xori %8, %extracted_slice_10 : tensor<4xi8>
      %inserted_slice_20 = tensor.insert_slice %6 into %arg2[%2, 0] [1, 4] [1, 1] : tensor<4xi8> into tensor<11x16xi8>
      %inserted_slice_21 = tensor.insert_slice %7 into %inserted_slice_20[%2, 4] [1, 4] [1, 1] : tensor<4xi8> into tensor<11x16xi8>
      %inserted_slice_22 = tensor.insert_slice %8 into %inserted_slice_21[%2, 8] [1, 4] [1, 1] : tensor<4xi8> into tensor<11x16xi8>
      %inserted_slice_23 = tensor.insert_slice %9 into %inserted_slice_22[%2, 12] [1, 4] [1, 1] : tensor<4xi8> into tensor<11x16xi8>
      scf.yield %inserted_slice_23 : tensor<11x16xi8>
    } {lower = 10 : i64, upper = 10 : i64}
    return %1 : tensor<11x16xi8>
  }
}

