module {
  func.func @compute_intersection_tensor(%X: tensor<4xi32>, %Y: tensor<4xi32>, %r: tensor<4xi32>)
      -> tensor<4xi32> {
    // Create an initial tensor of shape 4 filled with zeros.
    %init = arith.constant dense<0> : tensor<4xi32>

    // Define index constants for 0, 1, 2, and 3.
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index

    // --- Unroll computation for index 0 ---
    %y0 = tensor.extract %Y[%c0] : tensor<4xi32>
    %r0 = tensor.extract %r[%c0] : tensor<4xi32>
    %x0 = tensor.extract %X[%c0] : tensor<4xi32>
    %diff0_0 = arith.subi %y0, %x0 : i32
    %x1 = tensor.extract %X[%c1] : tensor<4xi32>
    %diff0_1 = arith.subi %y0, %x1 : i32
    %x2 = tensor.extract %X[%c2] : tensor<4xi32>
    %diff0_2 = arith.subi %y0, %x2 : i32
    %x3 = tensor.extract %X[%c3] : tensor<4xi32>
    %diff0_3 = arith.subi %y0, %x3 : i32
    %prod0_0 = arith.muli %diff0_0, %diff0_1 : i32
    %prod0_1 = arith.muli %prod0_0, %diff0_2 : i32
    %prod0 = arith.muli %prod0_1, %diff0_3 : i32
    %d0 = arith.muli %r0, %prod0 : i32
    %res0 = tensor.insert %d0 into %init[%c0] : tensor<4xi32>

    // --- Unroll computation for index 1 ---
    %y1 = tensor.extract %Y[%c1] : tensor<4xi32>
    %r1 = tensor.extract %r[%c1] : tensor<4xi32>
    %x0_1 = tensor.extract %X[%c0] : tensor<4xi32>
    %diff1_0 = arith.subi %y1, %x0_1 : i32
    %x1_1 = tensor.extract %X[%c1] : tensor<4xi32>
    %diff1_1 = arith.subi %y1, %x1_1 : i32
    %x2_1 = tensor.extract %X[%c2] : tensor<4xi32>
    %diff1_2 = arith.subi %y1, %x2_1 : i32
    %x3_1 = tensor.extract %X[%c3] : tensor<4xi32>
    %diff1_3 = arith.subi %y1, %x3_1 : i32
    %prod1_0 = arith.muli %diff1_0, %diff1_1 : i32
    %prod1_1 = arith.muli %prod1_0, %diff1_2 : i32
    %prod1 = arith.muli %prod1_1, %diff1_3 : i32
    %d1 = arith.muli %r1, %prod1 : i32
    %res1 = tensor.insert %d1 into %res0[%c1] : tensor<4xi32>

    // --- Unroll computation for index 2 ---
    %y2 = tensor.extract %Y[%c2] : tensor<4xi32>
    %r2 = tensor.extract %r[%c2] : tensor<4xi32>
    %x0_2 = tensor.extract %X[%c0] : tensor<4xi32>
    %diff2_0 = arith.subi %y2, %x0_2 : i32
    %x1_2 = tensor.extract %X[%c1] : tensor<4xi32>
    %diff2_1 = arith.subi %y2, %x1_2 : i32
    %x2_2 = tensor.extract %X[%c2] : tensor<4xi32>
    %diff2_2 = arith.subi %y2, %x2_2 : i32
    %x3_2 = tensor.extract %X[%c3] : tensor<4xi32>
    %diff2_3 = arith.subi %y2, %x3_2 : i32
    %prod2_0 = arith.muli %diff2_0, %diff2_1 : i32
    %prod2_1 = arith.muli %prod2_0, %diff2_2 : i32
    %prod2 = arith.muli %prod2_1, %diff2_3 : i32
    %d2 = arith.muli %r2, %prod2 : i32
    %res2 = tensor.insert %d2 into %res1[%c2] : tensor<4xi32>

    // --- Unroll computation for index 3 ---
    %y3 = tensor.extract %Y[%c3] : tensor<4xi32>
    %r3 = tensor.extract %r[%c3] : tensor<4xi32>
    %x0_3 = tensor.extract %X[%c0] : tensor<4xi32>
    %diff3_0 = arith.subi %y3, %x0_3 : i32
    %x1_3 = tensor.extract %X[%c1] : tensor<4xi32>
    %diff3_1 = arith.subi %y3, %x1_3 : i32
    %x2_3 = tensor.extract %X[%c2] : tensor<4xi32>
    %diff3_2 = arith.subi %y3, %x2_3 : i32
    %x3_3 = tensor.extract %X[%c3] : tensor<4xi32>
    %diff3_3 = arith.subi %y3, %x3_3 : i32
    %prod3_0 = arith.muli %diff3_0, %diff3_1 : i32
    %prod3_1 = arith.muli %prod3_0, %diff3_2 : i32
    %prod3 = arith.muli %prod3_1, %diff3_3 : i32
    %d3 = arith.muli %r3, %prod3 : i32
    %res3 = tensor.insert %d3 into %res2[%c3] : tensor<4xi32>

    return %res3 : tensor<4xi32>
  }
}
