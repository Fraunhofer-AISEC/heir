func.func @main(%input: tensor<1024xi32>) -> i32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 1024 : index
    %zero = arith.constant 0 : i32
    %sum = scf.for %i = %c0 to %c10 step %c1 iter_args(%acc = %zero) -> (i32) {
        %elem = tensor.extract %input[%i] : tensor<1024xi32>
        %new_acc = arith.addi %acc, %elem : i32
        scf.yield %new_acc : i32
    }
    return %sum : i32
}

