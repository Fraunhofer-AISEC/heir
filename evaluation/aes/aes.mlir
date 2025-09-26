module {
  // Vollständige S-Box für SubBytes

  func.func @aes_encrypt(%state: tensor<16xi8>) -> tensor<16xi8> {
    %sbox = arith.constant dense<"0x637C777BF26B6FC53001672BFED7AB76CA82C97DFA5947F0ADD4A2AF9CA472C0B7FD9326363FF7CC34A5E5F171D8311504C723C31896059A071280E2EB27B27509832C1A1B6E5AA0523BD6B329E32F8453D100ED20FCB15B6ACBBE394A4C58CFD0EFAAFB434D338545F9027F503C9FA851A3408F929D38F5BCB6DA2110FFF3D2CD0C13EC5F974417C4A77E3D645D197360814FDC222A908846EEB814DE5E0BDBE0323A0A4906245CC2D3AC629195E479E7C8376D8DD54EA96C56F4EA657AAE08BA78252E1CA6B4C6E8DD741F4BBD8B8A703EB5664803F60E613557B986C11D9EE1F8981169D98E949B1E87E9CE5528DF8CA1890DBFE6426841992D0FB054BB16"> : tensor<256xi8>
    // Platzhalter für Round Keys (hier nur Nullen, ersetzen durch echte Schlüssel)
    %round_keys = tensor.empty() : tensor<176xi8>

    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %sixteen = arith.constant 16 : index
    %ten = arith.constant 10 : index
    %t_empty = tensor.empty() : tensor<16xi8>

    // Initial AddRoundKey
    %state_in = scf.for %i = %zero to %sixteen step %one iter_args(%acc = %t_empty) -> tensor<16xi8> {
      %rk = tensor.extract %round_keys[%i] : tensor<176xi8>
      %b = tensor.extract %state[%i] : tensor<16xi8>
      %x = arith.xori %b, %rk : i8
      %acc_next = tensor.insert %x into %acc[%i] : tensor<16xi8>
      scf.yield %acc_next : tensor<16xi8>
    }

    // 9 Runden
    %state1 = scf.for %round = %one to %one step %one iter_args(%state_in_arg = %state_in) -> tensor<16xi8> {
      // SubBytes
      %subbed = scf.for %i = %zero to %sixteen step %one iter_args(%acc = %t_empty) -> tensor<16xi8> {
        %b = tensor.extract %state_in_arg[%i] : tensor<16xi8>
        %bi = arith.extui %b : i8 to i64
        %bii = arith.index_cast %bi : i64 to index
        %sb = tensor.extract %sbox[%bii] : tensor<256xi8>
        %acc_next = tensor.insert %sb into %acc[%i] : tensor<16xi8>
        scf.yield %acc_next : tensor<16xi8>
      }

      // AddRoundKey
      %rk_offset = arith.muli %round, %sixteen : index
      %next_state = scf.for %i = %zero to %sixteen step %one iter_args(%acc = %t_empty) -> tensor<16xi8> {
        %b = tensor.extract %subbed[%i] : tensor<16xi8>
        %rk_idx = arith.addi %rk_offset, %i : index
        %rk = tensor.extract %round_keys[%rk_idx] : tensor<176xi8>
        %x = arith.xori %b, %rk : i8
        %acc_next = tensor.insert %x into %acc[%i] : tensor<16xi8>
        scf.yield %acc_next : tensor<16xi8>
      }

      scf.yield %next_state : tensor<16xi8>
    }

    // Final Runde ohne MixColumns
    %final_r = scf.for %i = %zero to %sixteen step %one iter_args(%acc = %t_empty) -> tensor<16xi8> {
      %b = tensor.extract %state1[%i] : tensor<16xi8>
      %bi = arith.extui %b : i8 to i64
      %bii = arith.index_cast %bi : i64 to index
      %sb = tensor.extract %sbox[%bii] : tensor<256xi8>
      %rk_idx = arith.muli %ten, %sixteen : index
      %rk = tensor.extract %round_keys[%rk_idx] : tensor<176xi8>
      %x = arith.xori %sb, %rk : i8
      %acc_next = tensor.insert %x into %acc[%i] : tensor<16xi8>
      scf.yield %acc_next : tensor<16xi8>
    }

    func.return %final_r : tensor<16xi8>
  }
}