#!/bin/bash
# Record start time
script_start_time=$(date +%s)

if [ "$#" -eq 0 ]; then
    echo "Usage: $0 name:ciphertext_degree [name:ciphertext_degree ...] [group_name]" >&2
    echo "Available groups: basic, advanced, all" >&2
    exit 1
fi

# Check for GNU Parallel
if ! command -v parallel &> /dev/null; then
    echo "GNU Parallel is not installed. Installing it now..."
    sudo apt-get update && sudo apt-get install -y parallel
    
    if ! command -v parallel &> /dev/null; then
        echo "Failed to install GNU Parallel. Running in sequential mode."
        USE_PARALLEL=false
    else
        USE_PARALLEL=false
    fi
else
    USE_PARALLEL=true
fi

# Get number of available CPU cores
NUM_CORES=$(nproc)
if [ "$NUM_CORES" -gt 1 ]; then
    # Use 80% of available cores to avoid system overload
    NUM_JOBS=$(awk "BEGIN {print int($NUM_CORES * 0.8)}")
    if [ "$NUM_JOBS" -lt 1 ]; then
        NUM_JOBS=1
    fi
else
    NUM_JOBS=1
fi

echo "Running with $NUM_JOBS parallel jobs on $NUM_CORES detected CPU cores"

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
        rotate)
            echo "rotate-0:8 rotate-1:8 rotate-2:8 rotate-3:8 rotate-4:8"
            ;;
        form)
            echo "form-equal:8 form-high-low:8 form-low-high:8 form-hill:8 form-valley:8 form-increasing:8 form-decreasing:8"
            ;;
        add-eq)
            echo "add-eq-0:8 add-eq-1:8 add-eq-2:8 add-eq-3:8 add-eq-4:8 add-eq-5:8 add-eq-6:8"
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

# Function to handle command failures
command_failed() {
    echo "===============================================" >&2
    echo "ERROR: Command failed with exit code $1" >&2
    echo "Command: $2" >&2
    echo "===============================================" >&2
    return $1
}

# Ensure data directory exists
data_dir="$PWD/data"
if [ ! -d "$data_dir" ]; then
    echo "Creating data directory at $data_dir"
    mkdir -p "$data_dir"
fi

# Execute commands with error handling and track results
run_command() {
    echo "Running: $*" >&2
    local STDERR_FILE
    STDERR_FILE=$(mktemp)
    # Ensure pipefail is enabled for this command
    set -o pipefail
    "$@" 2> "$STDERR_FILE"
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        command_failed $exit_code "$*"
        cat "$STDERR_FILE" >&2
        rm "$STDERR_FILE"
        return $exit_code
    fi
    # Check if stderr contains common error words even if exit code is 0
    if grep -qi "error\|failed\|fatal\|exception" "$STDERR_FILE"; then
        echo "Command output contains error indicators:" >&2
        cat "$STDERR_FILE" >&2
        rm "$STDERR_FILE"
        return 1
    fi
    cat "$STDERR_FILE" >&2
    rm "$STDERR_FILE"
    return 0
}

# Function to print large headers
print_header() {
    # Redirect stdout to stderr for print_header function
    exec 3>&1  # Save the current stdout to file descriptor 3
    exec 1>&2  # Redirect stdout to stderr
    echo 
    echo "=================================================================="
    echo "==                                                              =="
    printf "==  %-59s ==\n" "$1"
    printf "==  %-59s ==\n" "$2"
    echo "==                                                              =="
    echo "=================================================================="
    echo
    # Now restore stdout back to its original state
    exec 1>&3  # Restore stdout from the saved file descriptor
    exec 3>&-  # Close the temporary file descriptor
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
}

# Function to process a single test
process_test() {
    local arg="$1"
    IFS=":" read -r name ciphertext_degree <<< "$arg"
    echo "Processing $name with ciphertext degree $ciphertext_degree..." >&2
    
    local TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    local test_result=0
    
    # Function to annotate parameters with a specific algorithm
    annotate_parameters() {
        local algorithm="$1"

        local input_mlir="$PWD/$name/$name-middle.mlir" 
        local output_mlir="$PWD/$name/$name-middle-params-${algorithm,,}.mlir"
        
        print_header "${algorithm^^} ALGORITHM" "Annotating parameters"
        
        TEMP_FILE=$(mktemp)

        if ! run_command bash -c "bazel run //tools:heir-opt -- --annotate-parameters=\"plaintext-modulus=65537 ring-dimension=0 algorithm=${algorithm^^}\" \"$input_mlir\" > \"$output_mlir\" 2> >(tee \"$TEMP_FILE\">&2 )"; then
            return 1
        fi

        if ! run_command extract_computed_params "$TEMP_FILE" "$PWD/data/${name}_${algorithm,,}_computed_$TIMESTAMP.json" "$name"; then
            return 1
        fi

        rm "$TEMP_FILE"

        if ! run_command python3 "$PWD/extract_bgv_params.py" $output_mlir "$PWD/data/${name}_${algorithm,,}_annotated_$TIMESTAMP.json" "$name" "${algorithm,,}"; then
            return 1
        fi

        print_header "${algorithm^^} ALGORITHM" "Validating noise with Mono model"
        if ! run_command bash -c "bazel run //tools:heir-opt -- --validate-noise=\"model=bgv-noise-mono\" \"$output_mlir\" --debug --debug-only=\"ValidateNoise\"" ; then
            return 1
        fi
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
        if ! run_command bazel run //tools:heir-opt -- --mlir-to-bgv="ciphertext-degree=${ciphertext_degree} insert-mgmt=false noise-model=bgv-noise-mono" --scheme-to-openfhe="entry-function=func" "$input_mlir" > "$output_mlir"; then
            return 1
        fi
    }

    # Function to generate OpenFHE code for a given implementation
    generate_openfhe_code() {
        local algorithm="$1"
        local input_mlir="$PWD/$name/$name-openfhe-${algorithm,,}.mlir"
        
        print_header "${algorithm^^} ALGORITHM" "Generating OpenFHE header"
        if ! run_command bazel run //tools:heir-translate -- --emit-openfhe-pke-header "$input_mlir" > "$name/${name}_${algorithm,,}.h"; then
            return 1
        fi
        
        print_header "${algorithm^^} ALGORITHM" "Generating OpenFHE implementation"
        if ! run_command bazel run //tools:heir-translate -- --emit-openfhe-pke "$input_mlir" > "$name/${name}_${algorithm,,}.cpp"; then
            return 1
        fi
    }

    # Function to run an implementation
    run_algorithm() {
        local algorithm="$1"
        
        print_header "EXECUTION" "Running ${algorithm^^} implementation"
        if ! run_command bazel run //mlir-test-files:main_${name}_${algorithm,,} -- ignoreComputation; then
            return 1
        fi
    }

    process_algorithm() {
        local algorithm="$1"
        local noParams="$2"
        
        if [ "$noParams" != "noParams" ]; then
            if ! annotate_parameters "$algorithm"; then
                return 1
            fi
        else 
            print_header "${algorithm^^} ALGORITHM" "Skipping parameter annotation"
        fi
        
        if ! convert_to_bgv_openfhe "$algorithm" "$noParams"; then
            return 1
        fi
        
        if ! generate_openfhe_code "$algorithm"; then
            return 1
        fi
    }

    run_gap_approach() {
        local model="$1"
        local model_name="${2:-$model}"  # Use second parameter as name or default to model
        
        print_header "GAP APPROACH" "Running ${model_name} model"
        if ! run_command bazel run //tools:heir-opt -- --debug --debug-only="GenerateParamBGV" --generate-param-bgv="model=${model}" $PWD/$name/$name-middle.mlir > $PWD/$name/$name-middle-params-gap-${model_name}.mlir; then
            return 1
        fi
        
        print_header "GAP APPROACH" "Extracting ${model_name} parameters"
        if ! run_command python3 "$PWD/extract_bgv_params.py" "$PWD/$name/$name-middle-params-gap-${model_name}.mlir" "$PWD/data/${name}_gap-${model_name}_annotated_$TIMESTAMP.json" "$name" "gap-${model_name}"; then
            return 1
        fi
    }
    
    print_header "PREPROCESSING" "Converting MLIR to secret arithmetic"
    if ! run_command bazel run //tools:heir-opt -- --secretize --mlir-to-secret-arithmetic --secret-insert-mgmt-bgv "$PWD/$name/$name.mlir" > "$PWD/$name/$name-middle.mlir"; then
        return 1
    fi

    # # Run gap approach with different noise models
    # if ! run_gap_approach "bgv-noise-by-bound-coeff-average-case" "kpz-avg"; then
    #     return 1
    # fi
    
    if ! run_gap_approach "bgv-noise-mono" "mono"; then
         return 1
    fi
    
    # Process each algorithm individually
    if ! process_algorithm "closed"; then
        return 1
    fi
    
    if ! process_algorithm "bisection"; then
        return 1
    fi
    
    if ! process_algorithm "balancing"; then
        return 1
    fi
    
    if ! process_algorithm "direct" "noParams"; then
        return 1
    fi

    print_header "POST-PROCESSING" "Fixing include paths in generated files"
    for algorithm in "bisection" "closed" "direct"; do
        if ! run_command sed -i 's|#include "openfhe/pke/openfhe.h"|#include "src/pke/include/openfhe.h" // from @openfhe|g' "$name/${name}_${algorithm,,}.h" "$name/${name}_${algorithm,,}.cpp"; then
            return 1
        fi
    done
    
    print_header "POST-PROCESSING" "Adding FIXEDAUTO scaling technique to direct variant"
    if ! run_command sed -i 's/CCParamsT params;/CCParamsT params;\n  params.SetScalingTechnique(FIXEDAUTO);/' "$name/${name}_direct.cpp"; then
        return 1
    fi
    
    # Run each implementation separately
    if ! run_algorithm "bisection"; then
        return 1
    fi
    
    if ! run_algorithm "closed"; then
        return 1
    fi
    
    if ! run_algorithm "direct"; then
        return 1
    fi
    
    return 0
}

# If not using parallel, run tests sequentially
if [ "$USE_PARALLEL" != "true" ]; then
    echo "Running tests sequentially..."
    failed_tests=()
    
    for arg in "${expanded_args[@]}"; do
        process_test "$arg"
        if [ $? -ne 0 ]; then
            failed_tests+=("$arg")
        fi
    done
else
    # Run tests in parallel
    echo "Running tests in parallel with $NUM_JOBS jobs..."
    
    # Create a temporary file to capture failures
    FAILURES_FILE=$(mktemp)
    # Export the functions so they can be used in parallel
    export -f process_test run_command command_failed print_header extract_computed_params
    
    # Run the tests in parallel with GNU Parallel
    printf "%s\n" "${expanded_args[@]}" | \
        parallel --jobs $NUM_JOBS --progress \
                 --results parallel_results \
                 "set -o pipefail; process_test {} || { echo {} >> $FAILURES_FILE; false; }"
    
    # Check for failures
    if [ -s "$FAILURES_FILE" ]; then
        failed_tests=()
        while read -r line; do
            failed_tests+=("$line")
        done < "$FAILURES_FILE"
    else
        failed_tests=()
    fi
    
    rm "$FAILURES_FILE"
fi

# Calculate and display elapsed time
script_end_time=$(date +%s)
elapsed_seconds=$((script_end_time - script_start_time))

# Format time nicely
if [ $elapsed_seconds -lt 60 ]; then
    echo -e "\nTotal execution time: ${elapsed_seconds} seconds"
else
    elapsed_minutes=$((elapsed_seconds / 60))
    elapsed_seconds=$((elapsed_seconds % 60))
    if [ $elapsed_minutes -lt 60 ]; then
        echo -e "\nTotal execution time: ${elapsed_minutes} minutes and ${elapsed_seconds} seconds"
    else
        elapsed_hours=$((elapsed_minutes / 60))
        elapsed_minutes=$((elapsed_minutes % 60))
        echo -e "\nTotal execution time: ${elapsed_hours} hours, ${elapsed_minutes} minutes and ${elapsed_seconds} seconds"
    fi
fi

# Report on failures
if [ ${#failed_tests[@]} -gt 0 ]; then
    echo -e "\n⚠️ Some tests failed:"
    for test in "${failed_tests[@]}"; do
        echo "  - $test"
    done
    exit 1
else
    echo -e "\n✅ All tests completed successfully"
fi
