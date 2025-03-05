#!/bin/bash
if [ "$#" -eq 0 ]; then
    echo "Usage: $0 name:ciphertext_degree [name:ciphertext_degree ...]" >&2
    exit 1
fi
set -e  # Exit immediately if any command fails

for arg in "$@"; do
    IFS=":" read -r name ciphertext_degree <<< "$arg"
    echo "Processing $name with ciphertext degree $ciphertext_degree..." >&2

    # Function to handle command failures
    command_failed() {
        echo "===============================================" >&2
        echo "ERROR: Command failed with exit code $1" >&2
        echo "Command: $2" >&2
        echo "===============================================" >&2
        exit $1
    }

    # Execute commands with error handling
    run_command() {
        echo "Running: $*" >&2
        "$@" || command_failed $? "$*"
    }

    run_command bazel run //tools:heir-opt -- --secretize --mlir-to-secret-arithmetic --secret-insert-mgmt-bgv "$PWD/$name/$name.mlir" > "$PWD/$name/$name-middle.mlir"
    
    # First run with BISECTION algorithm
    run_command bazel run //tools:heir-opt -- --annotate-parameters="plaintext-modulus=65539 ring-dimension=0 algorithm=BISECTION" "$PWD/$name/$name-middle.mlir" > "$PWD/$name/$name-middle-params-bisection.mlir"
    run_command bazel run //tools:heir-opt -- --mlir-to-bgv="ciphertext-degree=${ciphertext_degree} insert-mgmt=false"  --mlir-print-ir-after-all --scheme-to-openfhe="entry-function=func" "$PWD/$name/$name-middle-params-bisection.mlir" > "$PWD/$name/$name-openfhe-bisection.mlir"

    run_command bazel run //tools:heir-translate -- --emit-openfhe-pke-header "$PWD/$name/${name}-openfhe-bisection.mlir" > "$name/${name}_bisection.h"
    run_command bazel run //tools:heir-translate -- --emit-openfhe-pke "$PWD/$name/${name}-openfhe-bisection.mlir" > "$name/${name}_bisection.cpp"

    # Second run with OpenFHE algorithm
    run_command bazel run //tools:heir-opt -- --annotate-parameters="plaintext-modulus=65539 ring-dimension=0 algorithm=OpenFHE" "$PWD/$name/$name-middle.mlir" > "$PWD/$name/$name-middle-params-openfhe.mlir"
    run_command bazel run //tools:heir-opt -- --mlir-to-bgv="ciphertext-degree=${ciphertext_degree} insert-mgmt=false"  --mlir-print-ir-after-all --scheme-to-openfhe="entry-function=func" "$PWD/$name/$name-middle-params-openfhe.mlir" > "$PWD/$name/$name-openfhe-openfhe.mlir"

    run_command bazel run //tools:heir-translate -- --emit-openfhe-pke-header "$PWD/$name/${name}-openfhe-openfhe.mlir" > "$name/${name}_openfhe.h"
    run_command bazel run //tools:heir-translate -- --emit-openfhe-pke "$PWD/$name/${name}-openfhe-openfhe.mlir" > "$name/${name}_openfhe.cpp"
    
    # Third run with direct OpenFHE (no parameter annotation)
    run_command bazel run //tools:heir-opt -- --mlir-to-bgv="ciphertext-degree=${ciphertext_degree} insert-mgmt=false"  --mlir-print-ir-after-all --scheme-to-openfhe="entry-function=func" "$PWD/$name/$name-middle.mlir" > "$PWD/$name/$name-openfhe-direct.mlir"
    
    run_command bazel run //tools:heir-translate -- --emit-openfhe-pke-header "$PWD/$name/${name}-openfhe-direct.mlir" > "$name/${name}_direct.h"
    run_command bazel run //tools:heir-translate -- --emit-openfhe-pke "$PWD/$name/${name}-openfhe-direct.mlir" > "$name/${name}_direct.cpp"

    # Replace the include path in the generated C++ files
    run_command sed -i 's|#include "openfhe/pke/openfhe.h"|#include "src/pke/include/openfhe.h" // from @openfhe|g' "$name/${name}_bisection.h" "$name/${name}_bisection.cpp" "$name/${name}_openfhe.h" "$name/${name}_openfhe.cpp" "$name/${name}_direct.h" "$name/${name}_direct.cpp"
    
    # General command for adding noise eval outputs
    # run_command sed -i 's/^\(\s*\)const auto& \(ct[0-9][0-9]*\) = cc->\(EvalAdd\|EvalRotate\|ModReduce\)\(.*;\)$/\0\n\1std::cout << "\3 operation produced \2" << std::endl;/' /home/ubuntu/heir-aisec/mlir-test-files/${name/${name}.cpp

    # Run all versions
    run_command bazel run //mlir-test-files:main_${name}_bisection
    run_command bazel run //mlir-test-files:main_${name}_openfhe
    run_command bazel run //mlir-test-files:main_${name}_direct
done
