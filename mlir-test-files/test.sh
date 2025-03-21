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
        add-depth)
            echo "add-depth-0:8 add-depth-1:8 add-depth-2:8 add-depth-3:8 add-depth-4:8 add-depth-5:8 add-depth-6:8 add-depth-7:8 add-depth-8:8 add-depth-9:8 add-depth-10:8"
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

    TIMESTAMP=$(date +%Y%m%d_%H%M%S)

    # Function to print large headers
    print_header() {
        echo
        echo "=================================================================="
        echo "==                                                              =="
        echo "==  $1"
        printf "==  %-60s ==\n" "$2"
        echo "==                                                              =="
        echo "=================================================================="
        echo
    }

    extract_computed_params() {
        local input_file="$1"
        local output_file="$2"
        local test_name="$3"
        
        # Clean and extract the JSON content between <params> tags
        cat "$input_file" | tr -cd '\11\12\15\40-\176' | grep -zo "<params>.*</params>" | \
            sed 's/<params>//g' | sed 's/<\/params>//g' | \
            tr -d '\000-\011\013\014\016-\037\177' | \
            sed "s/<testname>/$test_name/g" > "$output_file"
            
        # Forward the original stderr output
        cat "$input_file" >&2
    }
    
    # Function to annotate parameters with a specific algorithm
    annotate_parameters() {
        local algorithm="$1"

        local input_mlir="$PWD/$name/$name-middle.mlir" 
        local output_mlir="$PWD/$name/$name-middle-params-${algorithm,,}.mlir"
        
        print_header "${algorithm^^} ALGORITHM" "Annotating parameters"
        
        TEMP_FILE=$(mktemp)
        run_command bazel run //tools:heir-opt -- --annotate-parameters="plaintext-modulus=65537 ring-dimension=0 algorithm=${algorithm^^}" "$input_mlir" > "$output_mlir" 2> >(tee "$TEMP_FILE" >&2)
        extract_computed_params "$TEMP_FILE" "$PWD/data/${name}_${algorithm,,}_computed_$TIMESTAMP.json" "$name"
        cat "$TEMP_FILE" >&2
        rm "$TEMP_FILE"

        run_command python3 "$PWD/extract_bgv_params.py" $output_mlir "$PWD/data/${name}_${algorithm,,}_annotated_$TIMESTAMP.json" "$name" "${algorithm,,}"

        print_header "${algorithm^^} ALGORITHM" "Validating noise with Mono model"
        run_command bazel run //tools:heir-opt -- --validate-noise="model=bgv-noise-mono" "$output_mlir" #--debug --debug-only="ValidateNoise"
    }
    
    # Function to convert to BGV and OpenFHE
    convert_to_bgv_openfhe() {
        local algorithm="$1"
        local noParams="$2"

        if [ "$noParams" == "noParams" ]; then
            local input_mlir="$PWD/$name/$name-middle.mlir"
        else
            local input_mlir="$PWD/$name/$name-middle-params-${algorithm,,}.mlir"
        fi
        local output_mlir="$PWD/$name/$name-openfhe-${algorithm,,}.mlir"

        print_header "${algorithm^^} ALGORITHM" "Converting to BGV and OpenFHE"
        run_command bazel run //tools:heir-opt -- --mlir-to-bgv="ciphertext-degree=${ciphertext_degree} insert-mgmt=false noise-model=bgv-noise-mono" --scheme-to-openfhe="entry-function=func" "$input_mlir" > "$output_mlir"
    }

    # Function to generate OpenFHE code for a given implementation
    generate_openfhe_code() {
        local algorithm="$1"
        local input_mlir="$PWD/$name/$name-openfhe-${algorithm,,}.mlir"
        
        print_header "${algorithm^^} ALGORITHM" "Generating OpenFHE header"
        run_command bazel run //tools:heir-translate -- --emit-openfhe-pke-header "$input_mlir" > "$name/${name}_${algorithm,,}.h"
        
        print_header "${algorithm^^} ALGORITHM" "Generating OpenFHE implementation"
        run_command bazel run //tools:heir-translate -- --emit-openfhe-pke "$input_mlir" > "$name/${name}_${algorithm,,}.cpp"
    }

    # Function to run an implementation
    run_algorithm() {
        local algorithm="$1"
        
        print_header "EXECUTION" "Running ${algorithm^^} implementation"
        run_command bazel run //mlir-test-files:main_${name}_${algorithm,,} -- ignoreComputation
    }

    process_algorithm() {
        local algorithm="$1"
        local noParams="$2"
        
        if [ "$noParams" != "noParams" ]; then
            annotate_parameters "$algorithm"
        else 
            print_header "${algorithm^^} ALGORITHM" "Skipping parameter annotation"
        fi
        convert_to_bgv_openfhe "$algorithm" "$noParams"
        generate_openfhe_code "$algorithm"
    }

    run_gap_approach() {
        local model="$1"
        local model_name="${2:-$model}"  # Use second parameter as name or default to model
        
        print_header "GAP APPROACH" "Running ${model_name} model"
        run_command bazel run //tools:heir-opt -- --generate-param-bgv="model=${model}" $PWD/$name/$name-middle.mlir > $PWD/$name/$name-middle-params-gap-${model_name}.mlir
        
        print_header "GAP APPROACH" "Extracting ${model_name} parameters"
        run_command python3 "$PWD/extract_bgv_params.py" "$PWD/$name/$name-middle-params-gap-${model_name}.mlir" "$PWD/data/${name}_gap-${model_name}_annotated_$TIMESTAMP.json" "$name" "gap-${model_name}"
    }
    
    print_header "PREPROCESSING" "Converting MLIR to secret arithmetic"
    run_command bazel run //tools:heir-opt -- --secretize --mlir-to-secret-arithmetic --secret-insert-mgmt-bgv "$PWD/$name/$name.mlir" > "$PWD/$name/$name-middle.mlir"

    # Run gap approach with different noise models
    run_gap_approach "bgv-noise-by-bound-coeff-avg-case" "kpz-avg"
    run_gap_approach "bgv-noise-mono" "mono"
    
    # Run the bisection algorithm pipeline
    process_algorithm "closed"
    process_algorithm "bisection"
    process_algorithm "balancing"
    process_algorithm "direct" noParams

    print_header "POST-PROCESSING" "Fixing include paths in generated files"
    for algorithm in "bisection" "closed" "direct"; do
        run_command sed -i 's|#include "openfhe/pke/openfhe.h"|#include "src/pke/include/openfhe.h" // from @openfhe|g' "$name/${name}_${algorithm,,}.h" "$name/${name}_${algorithm,,}.cpp"
    done
    
    print_header "POST-PROCESSING" "Adding FIXEDAUTO scaling technique to direct variant"
    run_command sed -i 's/CCParamsT params;/CCParamsT params;\n  params.SetScalingTechnique(FIXEDAUTO);/' "$name/${name}_direct.cpp"
    
    # General command for adding noise eval outputs
    # run_command sed -i 's/^\(\s*\)const auto& \(ct[0-9][0-9]*\) = cc->\(EvalAdd\|EvalRotate\|ModReduce\)\(.*;\)$/\0\n\1std::cout << "\3 operation produced \2" << std::endl;/' /home/ubuntu/heir-aisec/mlir-test-files/${name/${name}.cpp

    # Run all implementations
    run_algorithm "bisection"
    run_algorithm "closed"
    run_algorithm "direct"
done
