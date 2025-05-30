#!/bin/bash
# Record start time
script_start_time=$(date +%s)

# Global configuration
PLAINTEXT_MODULUS=65537 
export PLAINTEXT_MODULUS

# Define test-specific plaintext moduli
TEST_PLAINTEXT_MODULI=(
    "add-64-0:786433"
    "add-64-1:786433"
    "add-64-2:786433"
    "add-64-3:786433"
    "add-64-4:786433"
    "add-large-0:12582929"
    "add-large-5:12582929"
)

# Get plaintext modulus for a specific test
get_test_plaintext_modulus() {
    local test_name="$1"
    
    # Loop through the array to find the modulus
    for entry in "${TEST_PLAINTEXT_MODULI[@]}"; do
        IFS=':' read -r name value <<< "$entry"
        if [[ "$name" == "$test_name" ]]; then
            echo "$value"
            return
        fi
    done
    echo "$PLAINTEXT_MODULUS"  # Return default if not found
}

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
        USE_PARALLEL=true
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
            echo "add-eq-0:8 add-eq-2:8 add-eq-4:8 add-eq-6:8 add-eq-8:8 add-eq-10:8 add-eq-12:8 add-eq-14:8 add-eq-16:8 add-eq-18:8 add-eq-20:8"
            ;;
        add-num)
            echo "add-num-1:8 add-num-32:8 add-num-64:8 add-num-128:8"
            ;;
        rotate-num)
            echo "rotate-num-1:8 rotate-num-32:8 rotate-num-64:8 rotate-num-128:8"
            ;;
        rotate-of)
            echo "rotate-of-1:8 rotate-of-2:8 rotate-of-3:8 rotate-of-4:8"
            ;;
        all)
            # Combine all test groups
            local all_tests=""
            all_tests+=" $(get_group_tests add-64)"
            all_tests+=" $(get_group_tests add-depth)"
            all_tests+=" $(get_group_tests rotate)"
            all_tests+=" $(get_group_tests form)"
            all_tests+=" $(get_group_tests add-eq)"
            all_tests+=" $(get_group_tests add-num)"
            all_tests+=" $(get_group_tests rotate-num)"
            all_tests+=" $(get_group_tests rotate-of)"
            echo "$all_tests"
            ;;
    esac
    return 0
}

# Process arguments, expanding any groups and attaching plaintext moduli
expanded_args=()
for arg in "$@"; do
    # Check if this is a group name (no colon)
    if [[ "$arg" != *":"* ]]; then
        group_tests=$(get_group_tests "$arg")
        if [ $? -eq 0 ]; then
            # Add each test from the group to expanded args
            for group_arg in $group_tests; do
                # Get the test name part (before the colon)
                test_name="${group_arg%%:*}"
                # Get the ciphertext degree part (after the colon)
                ciphertext_degree="${group_arg#*:}"
                # Get the plaintext modulus for this test
                plaintext_mod=$(get_test_plaintext_modulus "$test_name")
                # Construct a new argument with plaintext modulus embedded
                expanded_args+=("${test_name}:${ciphertext_degree}:${plaintext_mod}")
            done
            echo "Expanded group '$arg' to: $group_tests" >&2
        else
            echo "Warning: '$arg' is not a valid group name or test specification (missing colon)" >&2
            echo "It should be either a group name or test_name:ciphertext_degree" >&2
        fi
    else
        # Regular test specification
        # Get the test name part (before the colon)
        test_name="${arg%%:*}"
        # Get the ciphertext degree part (after the colon)
        ciphertext_degree="${arg#*:}"
        # Get the plaintext modulus for this test
        plaintext_mod=$(get_test_plaintext_modulus "$test_name")
        # Construct a new argument with plaintext modulus embedded
        expanded_args+=("${test_name}:${ciphertext_degree}:${plaintext_mod}")
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
    local STDOUT_FILE
    STDOUT_FILE=$(mktemp)
    
    # Get calling function name or "main" if at top level
    local caller_func="${FUNCNAME[1]:-main}"
    
    # Get current test name and ensure test logs directory exists
    local test_id="${current_test_name:-unknown}"
    local test_log_dir="$PWD/$test_id/logs"
    if [ ! -d "$test_log_dir" ]; then
        mkdir -p "$test_log_dir"
    fi
    
    # Create log file paths for master log, stdout log, and stderr log
    local master_log="$test_log_dir/execution_${TIMESTAMP}.log"
    local stdout_log="$test_log_dir/stdout_${TIMESTAMP}.log"
    local stderr_log="$test_log_dir/stderr_${TIMESTAMP}.log"
    
    # Add a section divider to the master log
    {
        echo "=================================================================="
        echo "COMMAND: $*"
        echo "TIME: $(date)"
        echo "CALLER: $caller_func"
        echo "=================================================================="
    } >> "$master_log"
    
    # Ensure pipefail is enabled for this command
    set -o pipefail
    
    # Run command, capturing stdout and stderr separately
    # tee to console and to separate log files
    "$@" > >(tee -a "$STDOUT_FILE" "$stdout_log") 2> >(tee -a "$STDERR_FILE" "$stderr_log" >&2)
    local exit_code=$?
    
    # Add stdout and stderr to the master log file
    {
        echo
        echo "STDOUT:"
        echo "----------------------------------------------------------------"
        cat "$STDOUT_FILE"
        echo
        echo "STDERR:"
        echo "----------------------------------------------------------------"
        cat "$STDERR_FILE"
        echo
        echo "EXIT CODE: $exit_code"
        echo
    } >> "$master_log"
    
    if [ $exit_code -ne 0 ]; then
        command_failed $exit_code "$*"
        rm "$STDERR_FILE" "$STDOUT_FILE"
        return $exit_code
    fi
    
    # Check if stderr contains common error words even if exit code is 0
    if grep -qi "error\|failed\|fatal\|exception" "$STDERR_FILE"; then
        echo "Command output contains error indicators:" >&2
        cat "$STDERR_FILE" >&2
        {
            echo "WARNING: Command output contains error indicators but returned success code"
        } >> "$master_log"
        rm "$STDERR_FILE" "$STDOUT_FILE"
        return 1
    fi
    
    rm "$STDERR_FILE" "$STDOUT_FILE"
    return 0
}

# Function to print large headers
print_header() {
    # Save the header message to a temporary file for logging
    local header_file
    header_file=$(mktemp)
    
    {
        echo 
        echo "=================================================================="
        echo "==                                                              =="
        printf "==  %-59s ==\n" "$1"
        printf "==  %-59s ==\n" "$2"
        echo "==                                                              =="
        echo "=================================================================="
        echo
    } | tee "$header_file"

    # Get current test name and ensure test logs directory exists
    local test_id="${current_test_name:-unknown}"
    local test_log_dir="$PWD/$test_id/logs"
    if [ ! -d "$test_log_dir" ]; then
        mkdir -p "$test_log_dir"
    fi
    
    # Add the header to stdout and stderr logs
    if [[ -n "$TIMESTAMP" ]]; then
        cat "$header_file" >> "$test_log_dir/stdout_${TIMESTAMP}.log"
        cat "$header_file" >> "$test_log_dir/stderr_${TIMESTAMP}.log"
    fi
    
    rm "$header_file"
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
    # Split by colon to get test name, ciphertext degree, and plaintext modulus
    IFS=":" read -r name ciphertext_degree test_plaintext_modulus <<< "$arg"
    
    echo "Processing $name with ciphertext degree $ciphertext_degree and plaintext modulus $test_plaintext_modulus..." >&2
    
    # Set global test name for logging purposes
    current_test_name="$name"
    
    local TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    local test_result=0
    
    # Ensure test directory exists
    if [ ! -d "$PWD/$name" ]; then
        mkdir -p "$PWD/$name"
    fi
    
    # Ensure test logs directory exists
    local test_log_dir="$PWD/$name/logs"
    if [ ! -d "$test_log_dir" ]; then
        mkdir -p "$test_log_dir"
    fi
    
    # Initialize the master log file with test information
    local master_log="$test_log_dir/execution_${TIMESTAMP}.log"
    {
        echo "===================================================================="
        echo "TEST: $name"
        echo "CIPHERTEXT DEGREE: $ciphertext_degree"
        echo "PLAINTEXT MODULUS: $test_plaintext_modulus"
        echo "START TIME: $(date)"
        echo "===================================================================="
        echo
    } > "$master_log"
    
    # Export timestamp for use in the run_command function
    export TIMESTAMP
    
    # Function to annotate parameters with a specific algorithm
    annotate_parameters() {
        local algorithm="$1"

        local input_mlir="$PWD/$name/$name-middle.mlir" 
        local output_mlir="$PWD/$name/$name-middle-params-$(echo "$algorithm" | tr '[:upper:]' '[:lower:]').mlir"
        
        print_header "$(echo "$algorithm" | tr '[:lower:]' '[:upper:]') ALGORITHM" "Annotating parameters"
        
        TEMP_FILE=$(mktemp)

        if ! run_command bash -c "bazel run //tools:heir-opt -- --annotate-parameters=\"plaintext-modulus=${test_plaintext_modulus} ring-dimension=0 algorithm=$(echo "$algorithm" | tr '[:upper:]' '[:lower:]')\" \"$input_mlir\" > \"$output_mlir\" 2> >(tee \"$TEMP_FILE\">&2 )"; then
            return 1
        fi

        if ! run_command extract_computed_params "$TEMP_FILE" "$PWD/data/${name}_$(echo "$algorithm" | tr '[:lower:]' '[:lower:]')_computed_$TIMESTAMP.json" "$name"; then
            return 1
        fi

        rm "$TEMP_FILE"

        if ! run_command python3 "$PWD/extract_bgv_params.py" "$output_mlir" "$PWD/data/${name}_$(echo "$algorithm" | tr '[:lower:]' '[:lower:]')_annotated_$TIMESTAMP.json" "$name" "$(echo "$algorithm" | tr '[:lower:]' '[:lower:]')"; then
            return 1
        fi

        print_header "$(echo "$algorithm" | tr '[:lower:]' '[:upper:]') ALGORITHM" "Validating noise with Mono model"
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
            local input_mlir="$PWD/$name/$name-middle-params-$(echo "$algorithm" | tr '[:upper:]' '[:lower:]').mlir"
        fi
        local output_mlir="$PWD/$name/$name-openfhe-$(echo "$algorithm" | tr '[:upper:]' '[:lower:]').mlir"

        print_header "$(echo "$algorithm" | tr '[:lower:]' '[:upper:]') ALGORITHM" "Converting to BGV and OpenFHE"
        if ! run_command bazel run //tools:heir-opt -- --mlir-to-bgv="ciphertext-degree=${ciphertext_degree} insert-mgmt=false noise-model=bgv-noise-mono" --scheme-to-openfhe="entry-function=func" "$input_mlir" > "$output_mlir"; then
            return 1
        fi
    }

    # Function to generate OpenFHE code for a given implementation
    generate_openfhe_code() {
        local algorithm="$1"
        local input_mlir="$PWD/$name/$name-openfhe-$(echo "$algorithm" | tr '[:upper:]' '[:lower:]').mlir"
        
        print_header "$(echo "$algorithm" | tr '[:lower:]' '[:upper:]') ALGORITHM" "Generating OpenFHE header"
        if ! run_command bazel run //tools:heir-translate -- --emit-openfhe-pke-header "$input_mlir" > "$name/${name}_$(echo "$algorithm" | tr '[:upper:]' '[:lower:]').h"; then
            return 1
        fi
        
        print_header "$(echo "$algorithm" | tr '[:lower:]' '[:upper:]') ALGORITHM" "Generating OpenFHE implementation"
        if ! run_command bazel run //tools:heir-translate -- --emit-openfhe-pke "$input_mlir" > "$name/${name}_$(echo "$algorithm" | tr '[:upper:]' '[:lower:]').cpp"; then
            return 1
        fi
    }

    # Function to run an implementation
    run_algorithm() {
        local algorithm="$1"
        
        print_header "EXECUTION" "Running $(echo "$algorithm" | tr '[:lower:]' '[:upper:]') implementation"
        if ! run_command bazel run //mlir-test-files:main_${name}_$(echo "$algorithm" | tr '[:upper:]' '[:lower:]') -- ignoreComputation; then
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
            print_header "$(echo "$algorithm" | tr '[:lower:]' '[:upper:]') ALGORITHM" "Skipping parameter annotation"
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
        if ! run_command bazel run //tools:heir-opt -- --debug --debug-only="GenerateParamBGV" --debug-only=NoiseAnalysis --generate-param-bgv="model=${model}  plaintext-modulus=${test_plaintext_modulus}" "$PWD/$name/$name-middle.mlir" > "$PWD/$name/$name-middle-params-gap-$(echo "$model_name" | tr '[:upper:]' '[:lower:]').mlir"; then
            return 1
        fi
        
        print_header "GAP APPROACH" "Extracting ${model_name} parameters"
        if ! run_command python3 "$PWD/extract_bgv_params.py" "$PWD/$name/$name-middle-params-gap-$(echo "$model_name" | tr '[:upper:]' '[:lower:]').mlir" "$PWD/data/${name}_gap-$(echo "$model_name" | tr '[:upper:]' '[:lower:]')_annotated_$TIMESTAMP.json" "$name" "gap-$(echo "$model_name" | tr '[:upper:]' '[:lower:]')"; then
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

    # Try to run gap approach with mono model
    if ! run_gap_approach "bgv-noise-mono" "mono"; then
        echo "WARNING: Gap approach with mono model failed, continuing with other steps..." >&2
        # Don't return failure here, continue with the rest of the processing
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
        if ! run_command sed -i 's|#include "openfhe/pke/openfhe.h"|#include "src/pke/include/openfhe.h" // from @openfhe|g' "$name/${name}_$(echo "$algorithm" | tr '[:upper:]' '[:lower:]').h" "$name/${name}_$(echo "$algorithm" | tr '[:upper:]' '[:lower:]').cpp"; then
            return 1
        fi
    done
    
    print_header "POST-PROCESSING" "Adding FIXEDAUTO scaling technique to direct variant"
    # Fix scaling technique
    if ! run_command sed -i "s/CCParamsT params;/CCParamsT params;\n  params.SetScalingTechnique(FIXEDAUTO);/" "$name/${name}_direct.cpp"; then
        return 1
    fi
    
    # Fix plaintext modulus separately - use test_plaintext_modulus instead of PLAINTEXT_MODULUS
    if ! run_command sed -i "s/params.SetPlaintextModulus([0-9]*);/params.SetPlaintextModulus(${test_plaintext_modulus});/" "$name/${name}_direct.cpp"; then
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
    export -f process_test run_command command_failed print_header extract_computed_params get_test_plaintext_modulus
    
    # Run the tests in parallel with GNU Parallel
    printf "%s\n" "${expanded_args[@]}" | \
        parallel --jobs $NUM_JOBS --progress \
                 --results parallel_results \
                 --env SERIALIZED_TEST_PLAINTEXT_MODULI \
                 "set -o pipefail; process_test {} || { echo {} >> $FAILURES_FILE; false; }"
    
    # Check for failures
    failed_tests=()
    if [ -s "$FAILURES_FILE" ]; then
        while read -r line; do
            failed_tests+=("$line")
        done < "$FAILURES_FILE"
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
    elapsed_seconds=$((elapsed_minutes % 60))
    if [ $elapsed_minutes -lt 60 ]; then
        echo -e "\nTotal execution time: ${elapsed_minutes} minutes and ${elapsed_seconds} seconds"
    else
        elapsed_hours=$((elapsed_minutes / 60))
        elapsed_minutes=$((elapsed_minutes % 60))
        echo -e "\nTotal execution time: ${elapsed_hours} hours, ${elapsed_minutes} minutes and ${elapsed_seconds} seconds"
    fi
fi

# Report on failures
if [ -n "${failed_tests[*]}" ] && [ ${#failed_tests[@]} -gt 0 ]; then
    echo -e "\n⚠️ Some tests failed:"
    for test in "${failed_tests[@]}"; do
        # Extract test name from the test specification (remove ciphertext degree)
        test_name=$(echo "$test" | cut -d':' -f1)
        
        # Find the most recent log file for this test
        log_dir="$PWD/$test_name/logs"
        if [ -d "$log_dir" ]; then
            latest_log=$(ls -t "$log_dir"/execution_*.log 2>/dev/null | head -1)
            if [ -n "$latest_log" ]; then
                echo "  - $test (log: $latest_log)"
            else
                echo "  - $test (no log file found)"
            fi
        else
            echo "  - $test (no log directory found)"
        fi
    done
    exit 1
else
    echo -e "\n✅ All tests completed successfully"
fi