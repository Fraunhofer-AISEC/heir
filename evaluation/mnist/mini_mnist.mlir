module {
  func.func @main(%input: tensor<25xi32> ,
                %layer1_weights: tensor<15x25xi32>,
                %layer1_bias: tensor<15xi32>,
                %layer2_weights: tensor<10x15xi32>,
                %layer2_bias: tensor<10xi32>) -> tensor<15xi32> {
    // Konstanten als Variablen
    %c_0_idx = arith.constant 0 : index
    %c_one = arith.constant 1 : index
    %c_0_i16 = arith.constant 0 : i16
    %c_0_i32 = arith.constant 0 : i32
    %c_15 = arith.constant 15 : index
    %c_25 = arith.constant 25 : index
    %c_10 = arith.constant 10 : index

    // Output für erste Schicht
    %layer1_empty = tensor.empty() : tensor<15xi32>

    %layer1_out = scf.for %i = %c_0_idx to %c_15 step %c_one iter_args(%out = %layer1_empty) -> tensor<15xi32> {
      // Matrix-Vektor-Multiplikation
      %acc_final = scf.for %j = %c_0_idx to %c_25 step %c_one iter_args(%acc = %c_0_i32) -> (i32) {
        %weight = tensor.extract %layer1_weights[%i, %j] : tensor<15x25xi32>
        %input_val = tensor.extract %input[%j] : tensor<25xi32>
        %prod = arith.muli %weight, %input_val : i32
        %acc_next = arith.addi %acc, %prod : i32
        scf.yield %acc_next : i32
      }

      // Bias hinzufügen
      //%bias = tensor.extract %layer1_bias[%i] : tensor<15xi32>
      //%sum_bias = arith.addi %acc_final, %bias : i32

      // ReLU
      //%relu_mask = arith.cmpi sgt, %sum_bias, %c_0_i32 : i32
      //%relu_value = arith.select %relu_mask, %sum_bias, %c_0_i32 : i32

      // Truncieren auf i8
      //%out_next = tensor.insert %relu_value into %out[%i] : tensor<15xi32>
      scf.yield %out_next : tensor<15xi32>
    }

    func.return %layer1_out : tensor<15xi32>
  }
}