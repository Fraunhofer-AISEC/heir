func.func @approx_sign(%x: tensor<1x1024xi64>) -> tensor<1x1024xi64> {
  %c11 = arith.constant dense<-17023488> : tensor<1x1024xi64> // -260.03867215588 * 2^16
  %c9 = arith.constant dense<48940> : tensor<1x1024xi64> // 746.781707684981 * 2^16
  %c7 = arith.constant dense<-52481> : tensor<1x1024xi64> // -797.090149675776 * 2^16
  %c5 = arith.constant dense<25400> : tensor<1x1024xi64> // 388.964712077092 * 2^16
  %c3 = arith.constant dense<-566> : tensor<1x1024xi64> // -86.6415008377027 * 2^16
  %c1 = arith.constant dense<576> : tensor<1x1024xi64> // 8.82341343192733 * 2^16

  %x2 = arith.muli %x, %x : tensor<1x1024xi64>
  %x3 = arith.muli %x2, %x : tensor<1x1024xi64>
  %x4 = arith.muli %x2, %x2 : tensor<1x1024xi64>
  %x5 = arith.muli %x4, %x : tensor<1x1024xi64>
  %x6 = arith.muli %x4, %x2 : tensor<1x1024xi64>
  %x7 = arith.muli %x6, %x : tensor<1x1024xi64>
  %x8 = arith.muli %x4, %x4 : tensor<1x1024xi64>
  %x9 = arith.muli %x8, %x : tensor<1x1024xi64>
  %x11 = arith.muli %x5, %x6 : tensor<1x1024xi64>

  %s1 = arith.muli %x, %c1 : tensor<1x1024xi64>
  %s3 = arith.muli %x3, %c3 : tensor<1x1024xi64>
  %s5 = arith.muli %x5, %c5 : tensor<1x1024xi64>
  %s7 = arith.muli %x7, %c7 : tensor<1x1024xi64>
  %s9 = arith.muli %x9, %c9 : tensor<1x1024xi64>
  %s11 = arith.muli %x11, %c11 : tensor<1x1024xi64>

  %sum1 = arith.addi %s1, %s3 : tensor<1x1024xi64>
  %sum2 = arith.addi %sum1, %s5 : tensor<1x1024xi64>
  %sum3 = arith.addi %sum2, %s7 : tensor<1x1024xi64>
  %sum4 = arith.addi %sum3, %s9 : tensor<1x1024xi64>
  %sum5 = arith.addi %sum4, %s11 : tensor<1x1024xi64>
  return %sum5 : tensor<1x1024xi64>
}

func.func @approx_relu(%x: tensor<1x1024xi64>) -> tensor<1x1024xi64> {
  %sign = call @approx_sign(%x) : (tensor<1x1024xi64>) -> tensor<1x1024xi64>
  %signed = arith.muli %sign, %x : tensor<1x1024xi64>
  %sum = arith.addi %signed, %x : tensor<1x1024xi64>
  %c0_5 = arith.constant dense<32768> : tensor<1x1024xi64> // 0.5 * 2^16
  %norm = arith.muli %sum, %c0_5 : tensor<1x1024xi64>
  return %norm : tensor<1x1024xi64>
}

func.func @mlp(%input: tensor<1x1024xi64>, %fc1: tensor<1024x1024xi64>, %fc2: tensor<1024x1024xi64>, %fc1_buffer: tensor<1x1024xi64>, %fc2_buffer: tensor<1x1024xi64>) -> tensor<1x1024xi64> attributes {llvm.emit_c_interface} {
  %fc1_result = linalg.matmul ins(%input, %fc1 : tensor<1x1024xi64>, tensor<1024x1024xi64>) outs(%fc1_buffer : tensor<1x1024xi64>) -> tensor<1x1024xi64>

  %relu1 = call @approx_relu(%fc1_result) : (tensor<1x1024xi64>) -> tensor<1x1024xi64>

  %fc2_result = linalg.matmul ins(%relu1, %fc2 : tensor<1x1024xi64>, tensor<1024x1024xi64>) outs(%fc2_buffer : tensor<1x1024xi64>) -> tensor<1x1024xi64>

  return %fc2_result : tensor<1x1024xi64>
}
