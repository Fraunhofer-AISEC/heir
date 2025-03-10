#!/bin/bash
if [ "$#" -eq 0 ]; then
    echo "Usage: $0 name:ciphertext_degree [name:ciphertext_degree ...] [group_name]" >&2
    echo "Available groups: basic, advanced, all" >&2
    exit 1
fi
set -e  # Exit immediately if any command fails

# Define test groups
get_group_tests() {
    local group_name="$1"
    case "$group_name" in
        add-64)
            echo "add-64-0:8 add-64-1:8 add-64-2:8 add-64-3:8 add-64-4:8"
            ;;
    esac
    return 0
}

# Process arguments, expanding any groups
expanded_args=()
for arg in "$@"; do
    # Check if this is a group name (no colon)
    if [[ "$arg" != *":"* ]]; then
        group_tests=$(get_group_tests "$arg")
        if [ $? -eq 0 ]; then
            # Add each test from the group to expanded args
            for group_arg in $group_tests; do
                expanded_args+=("$group_arg")
            done
            echo "Expanded group '$arg' to: $group_tests" >&2
        else
            echo "Warning: '$arg' is not a valid group name or test specification (missing colon)" >&2
            echo "It should be either a group name or test_name:ciphertext_degree" >&2
        fi
    else
        # Regular test specification
        expanded_args+=("$arg")
    fi
done

# Run each test
for arg in "${expanded_args[@]}"; do
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
    # Capture the stderr and extract balancing results
    TEMP_FILE=$(mktemp)
    run_command bazel run //tools:heir-opt -- --annotate-parameters="plaintext-modulus=65539 ring-dimension=0 algorithm=BISECTION" "$PWD/$name/$name-middle.mlir" > "$PWD/$name/$name-middle-params-bisection.mlir" 2> "$TEMP_FILE"
    
    # Extract balancing results and save to JSON file in data directory with consistent naming
    # Ensure we clean non-printable characters and properly extract the JSON
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    cat "$TEMP_FILE" | tr -cd '\11\12\15\40-\176' | grep -zo "<balancing-result>.*</balancing-result>" | sed 's/<balancing-result>//g' | sed 's/<\/balancing-result>//g' | tr -d '\000-\011\013\014\016-\037\177' | sed "s/<testname>/$name/g" > "$PWD/data/${name}_balancing_$TIMESTAMP.json"
    cat "$TEMP_FILE" >&2  # Forward the original stderr output
    rm "$TEMP_FILE"
    
    run_command bazel run //tools:heir-opt -- --mlir-to-bgv="ciphertext-degree=${ciphertext_degree} insert-mgmt=false" --scheme-to-openfhe="entry-function=func" "$PWD/$name/$name-middle-params-bisection.mlir" > "$PWD/$name/$name-openfhe-bisection.mlir"

    run_command bazel run //tools:heir-translate -- --emit-openfhe-pke-header "$PWD/$name/${name}-openfhe-bisection.mlir" > "$name/${name}_bisection.h"
    run_command bazel run //tools:heir-translate -- --emit-openfhe-pke "$PWD/$name/${name}-openfhe-bisection.mlir" > "$name/${name}_bisection.cpp"

    # Second run with closed algorithm
    run_command bazel run //tools:heir-opt -- --annotate-parameters="plaintext-modulus=65539 ring-dimension=0 algorithm=CLOSED" "$PWD/$name/$name-middle.mlir" > "$PWD/$name/$name-middle-params-openfhe.mlir"
    run_command bazel run //tools:heir-opt -- --mlir-to-bgv="ciphertext-degree=${ciphertext_degree} insert-mgmt=false" --scheme-to-openfhe="entry-function=func" "$PWD/$name/$name-middle-params-openfhe.mlir" > "$PWD/$name/$name-openfhe-openfhe.mlir"

    run_command bazel run //tools:heir-translate -- --emit-openfhe-pke-header "$PWD/$name/${name}-openfhe-openfhe.mlir" > "$name/${name}_openfhe.h"
    run_command bazel run //tools:heir-translate -- --emit-openfhe-pke "$PWD/$name/${name}-openfhe-openfhe.mlir" > "$name/${name}_openfhe.cpp"
    
    # Third run with direct OpenFHE (no parameter annotation)
    run_command bazel run //tools:heir-opt -- --mlir-to-bgv="ciphertext-degree=${ciphertext_degree} insert-mgmt=false" --scheme-to-openfhe="entry-function=func" "$PWD/$name/$name-middle.mlir" > "$PWD/$name/$name-openfhe-direct.mlir"
    
    run_command bazel run //tools:heir-translate -- --emit-openfhe-pke-header "$PWD/$name/${name}-openfhe-direct.mlir" > "$name/${name}_direct.h"
    run_command bazel run //tools:heir-translate -- --emit-openfhe-pke "$PWD/$name/${name}-openfhe-direct.mlir" > "$name/${name}_direct.cpp"

    # Replace the include path in the generated C++ files
    run_command sed -i 's|#include "openfhe/pke/openfhe.h"|#include "src/pke/include/openfhe.h" // from @openfhe|g' "$name/${name}_bisection.h" "$name/${name}_bisection.cpp" "$name/${name}_openfhe.h" "$name/${name}_openfhe.cpp" "$name/${name}_direct.h" "$name/${name}_direct.cpp"
    
    # Add FIXEDAUTO scaling technique to direct variant
    run_command sed -i 's/CCParamsT params;/CCParamsT params;\n  params.SetScalingTechnique(FIXEDAUTO);/' "$name/${name}_direct.cpp"
    
    # General command for adding noise eval outputs
    # run_command sed -i 's/^\(\s*\)const auto& \(ct[0-9][0-9]*\) = cc->\(EvalAdd\|EvalRotate\|ModReduce\)\(.*;\)$/\0\n\1std::cout << "\3 operation produced \2" << std::endl;/' /home/ubuntu/heir-aisec/mlir-test-files/${name/${name}.cpp

    # Run all versions
    run_command bazel run //mlir-test-files:main_${name}_bisection
    run_command bazel run //mlir-test-files:main_${name}_openfhe
    run_command bazel run //mlir-test-files:main_${name}_direct
done
