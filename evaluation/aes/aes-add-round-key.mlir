module {
  // Vollständige S-Box für SubBytes

  func.func @aes_encrypt(%state: tensor<16xi8>, %round_keys : tensor<176xi8>) -> tensor<16xi8> {
    %sbox = arith.constant dense<"0x637C777BF26B6FC53001672BFED7AB76CA82C97DFA5947F0ADD4A2AF9CA472C0B7FD9326363FF7CC34A5E5F171D8311504C723C31896059A071280E2EB27B27509832C1A1B6E5AA0523BD6B329E32F8453D100ED20FCB15B6ACBBE394A4C58CFD0EFAAFB434D338545F9027F503C9FA851A3408F929D38F5BCB6DA2110FFF3D2CD0C13EC5F974417C4A77E3D645D197360814FDC222A908846EEB814DE5E0BDBE0323A0A4906245CC2D3AC629195E479E7C8376D8DD54EA96C56F4EA657AAE08BA78252E1CA6B4C6E8DD741F4BBD8B8A703EB5664803F60E613557B986C11D9EE1F8981169D98E949B1E87E9CE5528DF8CA1890DBFE6426841992D0FB054BB16"> : tensor<256xi8>

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

    func.return %state_in : tensor<16xi8>
  }
}