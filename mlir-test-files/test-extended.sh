#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if ! cd "$SCRIPT_DIR"; then
  echo "ERROR: Failed to change directory to $SCRIPT_DIR" >&2
  exit 1
fi

TEST_GROUP="${TEST_GROUP:-add-64}"
TEST_NAMES="${TEST_NAMES:-}"
LIST_TESTS_ONLY="${LIST_TESTS_ONLY:-0}"

PLAINTEXT_MODULUS="${PLAINTEXT_MODULUS:-786433}"
RING_DIMENSION="${RING_DIMENSION:-16384}"
CIPHERTEXT_DEGREE="${CIPHERTEXT_DEGREE:-1024}"
NOISE_MODEL="${NOISE_MODEL:-bgv-noise-mono}"
SKIP_NOISE_ANALYSIS="${SKIP_NOISE_ANALYSIS:-0}"
NON_FATAL_RUNTIME_CHECKS="${NON_FATAL_RUNTIME_CHECKS:-1}"
PARALLEL_JOBS="${PARALLEL_JOBS:-}"
DASHBOARD_REFRESH_SEC="${DASHBOARD_REFRESH_SEC:-10}"

OPENFHE_APPROACHES="${OPENFHE_APPROACHES:-direct closed bisection}"
LATTIGO_APPROACHES="${LATTIGO_APPROACHES:-greedy gap-mono}"
STOP_AFTER_PARAM_EXTRACTION="${STOP_AFTER_PARAM_EXTRACTION:-0}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SCRIPT_START_EPOCH="$(date +%s)"

source "$SCRIPT_DIR/lib/codegen_steps.sh"

declare -a ACTIVE_PARALLEL_PIDS=()
ABORT_CLEANUP_DONE=0

cleanup_parallel_workers() {
  local reason="${1:-abort}"
  local -a live_pids=()
  local pid

  for pid in "${ACTIVE_PARALLEL_PIDS[@]}"; do
    [[ -n "$pid" ]] || continue
    if kill -0 "$pid" 2>/dev/null; then
      live_pids+=("$pid")
    fi
  done

  if [[ ${#live_pids[@]} -eq 0 ]]; then
    return 0
  fi

  echo
  echo "Abort cleanup (${reason}): terminating ${#live_pids[@]} parallel worker(s)..." >&2

  for pid in "${live_pids[@]}"; do
    kill -TERM "$pid" 2>/dev/null || true
  done

  for pid in "${live_pids[@]}"; do
    wait "$pid" 2>/dev/null || true
  done

  for pid in "${live_pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -KILL "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  done

  ACTIVE_PARALLEL_PIDS=()
}

on_abort_signal() {
  local signal_name="$1"
  if [[ "$ABORT_CLEANUP_DONE" -eq 0 ]]; then
    ABORT_CLEANUP_DONE=1
    cleanup_parallel_workers "$signal_name"
  fi

  if [[ "$signal_name" == "INT" ]]; then
    exit 130
  fi
  if [[ "$signal_name" == "TERM" ]]; then
    exit 143
  fi
  exit 1
}

on_exit_cleanup() {
  local exit_code="$1"
  if [[ "$exit_code" -eq 0 ]]; then
    return
  fi
  if [[ "$ABORT_CLEANUP_DONE" -eq 0 ]]; then
    ABORT_CLEANUP_DONE=1
    cleanup_parallel_workers "EXIT($exit_code)"
  fi
}

trap 'on_abort_signal INT' INT
trap 'on_abort_signal TERM' TERM
trap 'on_exit_cleanup "$?"' EXIT

default_parallel_jobs() {
  local cpu_count
  cpu_count="$(nproc 2>/dev/null || echo 2)"
  if [[ -z "$cpu_count" || "$cpu_count" -lt 1 ]]; then
    cpu_count=2
  fi
  if [[ "$cpu_count" -gt 8 ]]; then
    cpu_count=8
  fi
  echo "$cpu_count"
}

if [[ -z "$PARALLEL_JOBS" ]]; then
  PARALLEL_JOBS="$(default_parallel_jobs)"
fi

if [[ "$PARALLEL_JOBS" -lt 1 ]]; then
  PARALLEL_JOBS=1
fi

if ! [[ "$DASHBOARD_REFRESH_SEC" =~ ^[0-9]+$ ]] || [[ "$DASHBOARD_REFRESH_SEC" -lt 1 ]]; then
  DASHBOARD_REFRESH_SEC=10
fi

build_tools() {
  echo "Building native HEIR tools once..."
  local build_start
  build_start="$(date +%s)"
  if ! (cd "$REPO_ROOT" && bazel build //tools:heir-opt //tools:heir-translate); then
    echo "ERROR: Failed to build required tools (heir-opt/heir-translate)." >&2
    exit 1
  fi
  local build_elapsed
  build_elapsed=$(( $(date +%s) - build_start ))
  echo "Build completed in $(format_duration "$build_elapsed") (${build_elapsed}s)."

  HEIR_OPT="$REPO_ROOT/bazel-bin/tools/heir-opt"
  HEIR_TRANSLATE="$REPO_ROOT/bazel-bin/tools/heir-translate"

  if [[ ! -x "$HEIR_OPT" ]] || [[ ! -x "$HEIR_TRANSLATE" ]]; then
    echo "ERROR: Expected native executables not found in bazel-bin." >&2
    echo "  HEIR_OPT=$HEIR_OPT" >&2
    echo "  HEIR_TRANSLATE=$HEIR_TRANSLATE" >&2
    exit 1
  fi
}

TESTS=()
FAILED_RUNTIME_CHECKS=()
FAILED_TESTS=()

append_unique_test() {
  local candidate="$1"
  local existing
  for existing in "${TESTS[@]}"; do
    if [[ "$existing" == "$candidate" ]]; then
      return
    fi
  done
  TESTS+=("$candidate")
}

emit_group_tests() {
  local group="$1"
  case "$group" in
    add-64)
      printf '%s\n' add-64-0 add-64-1 add-64-2 add-64-3 add-64-4
      ;;
    form)
      printf '%s\n' form-equal form-high-low form-low-high form-hill form-valley form-increasing form-decreasing
      ;;
    rotate)
      printf '%s\n' rotate-0 rotate-1 rotate-2 rotate-3 rotate-4
      ;;
    add-num)
      printf '%s\n' add-num-1 add-num-32 add-num-64 add-num-128
      ;;
    rotate-num)
      printf '%s\n' rotate-num-1 rotate-num-32 rotate-num-64 rotate-num-128
      ;;
    add-eq)
      # Requested subset for add-eq experiments.
      printf '%s\n' add-eq-0 add-eq-2 add-eq-6 add-eq-10 add-eq-14 add-eq-18
      ;;
    all)
      # Bundle of requested suites in this thread.
      printf '%s\n' \
        add-64-0 add-64-1 add-64-2 add-64-3 add-64-4 \
        form-equal form-high-low form-low-high form-hill form-valley form-increasing form-decreasing \
        rotate-0 rotate-1 rotate-2 rotate-3 rotate-4 \
        add-num-1 add-num-32 add-num-64 add-num-128 \
        rotate-num-1 rotate-num-32 rotate-num-64 rotate-num-128 \
        add-eq-0 add-eq-2 add-eq-6 add-eq-10 add-eq-14 add-eq-18
      ;;
    *)
      return 1
      ;;
  esac
}

resolve_selector() {
  local selector="$1"
  local -a _resolved_tests=()
  local test_name

  if emit_group_tests "$selector" >/dev/null 2>&1; then
    while IFS= read -r test_name; do
      [[ -n "$test_name" ]] || continue
      if [[ -f "$SCRIPT_DIR/$test_name/$test_name.mlir" ]]; then
        printf '%s\n' "$test_name"
      fi
    done < <(emit_group_tests "$selector")
    return 0
  fi

  if [[ -d "$SCRIPT_DIR/$selector" && -f "$SCRIPT_DIR/$selector/$selector.mlir" ]]; then
    printf '%s\n' "$selector"
    return 0
  fi

  mapfile -t _resolved_tests < <(
    find "$SCRIPT_DIR" -mindepth 1 -maxdepth 1 -type d -name "${selector}-*" -printf '%f\n' | sort
  )
  if [[ ${#_resolved_tests[@]} -eq 0 ]]; then
    return 1
  fi

  for test_name in "${_resolved_tests[@]}"; do
    if [[ -f "$SCRIPT_DIR/$test_name/$test_name.mlir" ]]; then
      printf '%s\n' "$test_name"
    fi
  done
}

discover_tests() {
  local -a selectors=()
  local selector
  local resolved

  if [[ $# -gt 0 ]]; then
    selectors=("$@")
  elif [[ -n "$TEST_NAMES" ]]; then
    IFS=',' read -r -a selectors <<< "$TEST_NAMES"
  else
    selectors=("$TEST_GROUP")
  fi

  for selector in "${selectors[@]}"; do
    selector="${selector// /}"
    [[ -n "$selector" ]] || continue

    while IFS= read -r resolved; do
      [[ -n "$resolved" ]] || continue
      append_unique_test "$resolved"
    done < <(resolve_selector "$selector" || true)
  done
}

group_family() {
  local test_name="$1"
  echo "${test_name%-*}"
}

record_runtime_failure() {
  local test_name="$1"
  local label="$2"
  FAILED_RUNTIME_CHECKS+=("${test_name}:${label}")
}

command_failed() {
  echo "===============================================" >&2
  echo "ERROR: Command failed with exit code $1" >&2
  echo "Command: $2" >&2
  echo "===============================================" >&2
  return "$1"
}

run_command() {
  echo "Running: $*" >&2
  local start_epoch
  start_epoch="$(date +%s)"
  if [[ -z "${MASTER_LOG:-}" ]]; then
    MASTER_LOG="/tmp/heir_test_extended_${TIMESTAMP}.log"
  fi
  if [[ -z "${LOG_DIR:-}" ]]; then
    LOG_DIR="/tmp"
  fi
  local stderr_file
  stderr_file=$(mktemp)
  local stdout_file
  stdout_file=$(mktemp)

  {
    echo "=================================================================="
    echo "COMMAND: $*"
    echo "TIME: $(date)"
    echo "=================================================================="
  } >> "$MASTER_LOG"

  set -o pipefail
  "$@" > >(tee -a "$stdout_file" "$LOG_DIR/stdout_${TIMESTAMP}.log") \
       2> >(tee -a "$stderr_file" "$LOG_DIR/stderr_${TIMESTAMP}.log" >&2)
  local exit_code=$?
    local end_epoch
    end_epoch="$(date +%s)"
    local elapsed_seconds=$((end_epoch - start_epoch))
    local elapsed_hms
    elapsed_hms="$(format_duration "$elapsed_seconds")"

  {
    echo
    echo "STDOUT:"
    echo "----------------------------------------------------------------"
    cat "$stdout_file"
    echo
    echo "STDERR:"
    echo "----------------------------------------------------------------"
    cat "$stderr_file"
    echo
    echo "EXIT CODE: $exit_code"
    echo "ELAPSED: ${elapsed_hms} (${elapsed_seconds}s)"
    echo
  } >> "$MASTER_LOG"

  echo "Elapsed: ${elapsed_hms} (${elapsed_seconds}s)" >&2

  if [[ $exit_code -ne 0 ]]; then
    command_failed "$exit_code" "$*"
    rm -f "$stderr_file" "$stdout_file"
    return "$exit_code"
  fi

  rm -f "$stderr_file" "$stdout_file"
  return 0
}

print_header() {
  echo >&2
  echo "==================================================================" >&2
  printf "==  %-59s ==\n" "$1" >&2
  printf "==  %-59s ==\n" "$2" >&2
  echo "==================================================================" >&2
  echo >&2
}

extract_computed_params() {
  local input_file="$1"
  local output_file="$2"
  local test_name="$3"
  awk '
    {
      if (!capture) {
        if (match($0, /<params>/)) {
          line = substr($0, RSTART + RLENGTH)
          if (match(line, /<\/params>/)) {
            print substr(line, 1, RSTART - 1)
            exit
          }
          print line
          capture = 1
        }
      } else {
        if (match($0, /<\/params>/)) {
          print substr($0, 1, RSTART - 1)
          exit
        }
        print
      }
    }
  ' "$input_file" | sed '/^[[:space:]]*$/d' | sed "s/<testname>/${test_name}/g" > "$output_file"
}

# extract_operation_counts_file() {
#   local input_file="$1"
#   local output_file="$2"
#   local test_name="$3"
#   local source_tag="$4"

#   run_command python3 "$OPCOUNT_TOOL" extract \
#     --input "$input_file" \
#     --output "$output_file" \
#     --test-name "$test_name" \
#     --source "$source_tag"
# }

# extract_operation_counts_for_test() {
#   local op_test_dir="$OPCOUNT_ROOT/$TEST_NAME"
#   mkdir -p "$op_test_dir"

#   local source
#   local approach
#   declare -A seen_sources=()
#   local approaches=()

#   read -r -a approaches <<< "$OPENFHE_APPROACHES $LATTIGO_APPROACHES"
#   for approach in "${approaches[@]}"; do
#     [[ -n "$approach" ]] || continue
#     seen_sources["$approach"]=1
#   done

#   for source in "${!seen_sources[@]}"; do
#     local annotate_log="$LOG_DIR/annotate_${source}_${TIMESTAMP}.log"
#     if [[ -f "$annotate_log" ]]; then
#       extract_operation_counts_file \
#         "$annotate_log" \
#         "$op_test_dir/${TEST_NAME}_${source}_operation_counts_${TIMESTAMP}.csv" \
#         "$TEST_NAME" \
#         "annotate-${source}"
#     fi
#   done

#   # local merged_log="$LOG_DIR/stderr_${TIMESTAMP}.log"
#   # if [[ -f "$merged_log" ]]; then
#   #   extract_operation_counts_file \
#   #     "$merged_log" \
#   #     "$op_test_dir/${TEST_NAME}_merged_operation_counts_${TIMESTAMP}.csv" \
#   #     "$TEST_NAME" \
#   #     "merged-stderr"
#   # fi
# }

to_lower() {
  echo "$1" | tr '[:upper:]' '[:lower:]'
}

fix_openfhe_include_paths() {
  # Detect if running on macOS for sed in-place flag compatibility.
  local sed_command
  if [[ "$(uname)" == "Darwin" ]]; then
    sed_command=(sed -i '')
  else
    sed_command=(sed -i)
  fi

  print_header "$TEST_NAME POST-PROCESSING" "Fixing include paths in generated OpenFHE files"

  local tag
  for tag in "$@"; do
    local lower_tag
    lower_tag="$(to_lower "$tag")"
    local header_file="$TEST_DIR/${TEST_NAME}_${lower_tag}.h"
    local impl_file="$TEST_DIR/${TEST_NAME}_${lower_tag}.cpp"

    if [[ ! -f "$header_file" || ! -f "$impl_file" ]]; then
      echo "WARNING: Skipping include path fix for '$tag' due to missing generated files." >&2
      continue
    fi

    if ! run_command "${sed_command[@]}" \
      's|#include "openfhe/pke/openfhe.h"|#include "src/pke/include/openfhe.h" // from @openfhe|g' \
      "$header_file" "$impl_file"; then
      echo "WARNING: Include path fix failed for '$tag'; continuing." >&2
    fi
  done
}

format_duration() {
  local total_seconds="$1"
  local hours=$((total_seconds / 3600))
  local minutes=$(((total_seconds % 3600) / 60))
  local seconds=$((total_seconds % 60))
  printf "%02d:%02d:%02d" "$hours" "$minutes" "$seconds"
}

annotate_algorithm() {
  local algorithm="$1"
  local lower
  lower="$(to_lower "$algorithm")"
  local output_mlir="$TEST_DIR/$TEST_NAME-middle-params-$lower.mlir"
  local annotate_stderr="$LOG_DIR/annotate_${lower}_${TIMESTAMP}.log"
  local params_json="$DATA_DIR/${TEST_NAME}_${lower}_computed_${TIMESTAMP}.json"
  local annotated_json="$DATA_DIR/${TEST_NAME}_${lower}_annotated_${TIMESTAMP}.json"

  print_header "$TEST_NAME $algorithm" "Annotating parameters"
  run_command "$HEIR_OPT" \
    "--annotate-parameters=plaintext-modulus=${PLAINTEXT_MODULUS} algorithm=${algorithm}" \
    "$MIDDLE_MLIR" \
    > "$output_mlir" 2> >(tee "$annotate_stderr" >&2)

  if [[ "$lower" != "direct" ]]; then
    print_header "$TEST_NAME $algorithm" "Extracting computed params"
    extract_computed_params "$annotate_stderr" "$params_json" "$TEST_NAME"
    if [[ ! -s "$params_json" ]]; then
      echo "WARNING: No <params> payload extracted for $algorithm in $TEST_NAME." >&2
    fi

    print_header "$TEST_NAME $algorithm" "Extracting annotated params"
    run_command bash -c 'python3 "$1" "$2" "$3" "$4" "$5" > /dev/null' _ \
      "$SCRIPT_DIR/extract_bgv_params.py" \
      "$output_mlir" \
      "$annotated_json" \
      "$TEST_NAME" \
      "$lower"

    if [[ "$SKIP_NOISE_ANALYSIS" == "1" ]]; then
      print_header "$TEST_NAME $algorithm" "Skipping noise validation (SKIP_NOISE_ANALYSIS=1)"
    else
      print_header "$TEST_NAME $algorithm" "Validating noise with Mono model"
      if ! run_command bash -c '"$1" "$2" "$3" > /dev/null' _\
        "$HEIR_OPT" \
        '--validate-noise=model=bgv-noise-mono' \
        "$output_mlir"; then
        echo "WARNING: Noise validation failed for $algorithm in $TEST_NAME; continuing." >&2
      fi
    fi
  fi

  echo "$output_mlir"
}

infer_lattigo_arity() {
  local input_go="$1"
  local helper_arity

  helper_arity="$(grep -cE '^func func__encrypt__arg[0-9]+\(' "$input_go" || true)"
  if [[ -n "$helper_arity" && "$helper_arity" -gt 0 ]]; then
    echo "$helper_arity"
    return 0
  fi

#   python3 - "$input_go" <<'PY'
# import pathlib
# import re
# import sys

# path = pathlib.Path(sys.argv[1])
# text = path.read_text()
# match = re.search(r'func\s+func\s*\(', text)
# if not match:
#   print(0)
#   raise SystemExit(0)

# i = match.end()
# depth = 1
# args = []
# while i < len(text) and depth > 0:
#   ch = text[i]
#   if ch == '(':
#     depth += 1
#   elif ch == ')':
#     depth -= 1
#     if depth == 0:
#       break
#   args.append(ch)
#   i += 1

# arg_text = ''.join(args)
# arity = len(re.findall(r'\*rlwe\.Ciphertext', arg_text))
# print(arity)
# PY
}

run_gap_mono() {
  print_header "$TEST_NAME GAP APPROACH" "Running mono model"

  run_command "$HEIR_OPT" \
    "--generate-param-bgv=model=bgv-noise-mono plaintext-modulus=${PLAINTEXT_MODULUS} slot-number=${CIPHERTEXT_DEGREE}" \
    "$MIDDLE_MLIR" \
    > "$GAP_MONO_MLIR"

  print_header "$TEST_NAME GAP APPROACH" "Extracting mono params"

  run_command python3 "$SCRIPT_DIR/extract_bgv_params.py" \
    "$GAP_MONO_MLIR" \
    "$DATA_DIR/${TEST_NAME}_gap-mono_annotated_${TIMESTAMP}.json" \
    "$TEST_NAME" \
    "gap-mono"
}

lower_to_bgv() {
  local input_mlir="$1"
  local tag="$2"
  local output_mlir="$TEST_DIR/$TEST_NAME-bgv-$tag.mlir"

  print_header "$TEST_NAME $tag" "Post-annotate lowering to BGV"
  
  run_command bash -c '"$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" "${10}" "${11}" "${12}" "${13}" "${14}" > "${15}"' _ \
    "$HEIR_OPT" \
    "--generate-param-bgv=model=${NOISE_MODEL} plaintext-modulus=${PLAINTEXT_MODULUS} slot-number=${CIPHERTEXT_DEGREE} use-public-key=true encryption-technique-extended=false" \
    --populate-scale-bgv \
    --canonicalize \
    --secret-distribute-generic \
    --canonicalize \
    "--secret-to-bgv=poly-mod-degree=${CIPHERTEXT_DEGREE}" \
    --forward-insert-to-extract \
    --canonicalize \
    --cse \
    --split-preprocessing=max-return-values=0 \
    "--convert-elementwise-to-affine=convert-dialects=ckks,bgv,lwe" \
    --tensor-ext-to-tensor \
    "$input_mlir" \
    "$output_mlir"

  echo "$output_mlir"
}

run_openfhe_checks() {
  local approaches
  read -r -a approaches <<< "$OPENFHE_APPROACHES"
  local params_only_mode=0

  if [[ "$STOP_AFTER_PARAM_EXTRACTION" == "1" ]]; then
    params_only_mode=1
  fi

  for approach in "${approaches[@]}"; do
    local target="//mlir-test-files:main_${TEST_NAME}_${approach}"
    local run_failed=0
    local params_output="$DATA_DIR/${TEST_NAME}_${approach}_execution_${TIMESTAMP}.json"

    if [[ "$params_only_mode" == "1" ]]; then
      print_header "$TEST_NAME OPENFHE $approach" "Extracting backend params (skipping encrypted computation)"
    else
      print_header "$TEST_NAME OPENFHE $approach" "Running encrypted computation + plaintext check"
    fi

    if ! (cd "$REPO_ROOT" && bazel query "$target" >/dev/null 2>&1); then
      echo "WARNING: Skipping $target (target not found)." >&2
      continue
    fi

    local run_cmd="cd \"$REPO_ROOT\" && HEIR_EXECUTION_PARAMS_FILE=\"$params_output\" bazel run \"$target\""
    if [[ "$params_only_mode" == "1" ]]; then
      run_cmd+=" -- ignoreComputation"
    fi

    if ! run_command bash -c "$run_cmd"; then
      run_failed=1
      record_runtime_failure "$TEST_NAME" "openfhe-${approach}"
      if [[ "$NON_FATAL_RUNTIME_CHECKS" != "1" ]]; then
        return 1
      fi
    fi

    if [[ ! -s "$params_output" ]]; then
      echo "WARNING: Missing OpenFHE execution params for $TEST_NAME/$approach at $params_output." >&2
      record_runtime_failure "$TEST_NAME" "openfhe-${approach}-missing-execution-params"
      if [[ "$NON_FATAL_RUNTIME_CHECKS" != "1" ]]; then
        return 1
      fi
    fi
  done
}

run_lattigo_check() {
  local tag="$1"
  local run_failed=0
  local params_only_mode=0

  if [[ "$STOP_AFTER_PARAM_EXTRACTION" == "1" ]]; then
    params_only_mode=1
  fi

  local input_go="$TEST_DIR/${TEST_NAME}_${tag}_lattigo.go"
  local normalized_go="$TEST_DIR/${TEST_NAME}_${tag}_lattigo_run.go"
  local runner_go="$TEST_DIR/${TEST_NAME}_${tag}_lattigo_main.go"
  local params_output="$DATA_DIR/${TEST_NAME}_${tag}_execution_${TIMESTAMP}.json"

  if [[ ! -f "$input_go" ]]; then
    echo "WARNING: Skipping Lattigo runtime check, missing generated file: $input_go" >&2
    return 1
  fi

  local arity
  arity="$(infer_lattigo_arity "$input_go")"
  if [[ -z "$arity" || "$arity" -eq 0 ]]; then
    echo "WARNING: Could not infer Lattigo input arity for $TEST_NAME/$tag." >&2
    return 1
  fi

  print_header "$TEST_NAME LATTIGO $tag" "Preparing runnable Go files"
  normalize_lattigo_file "$input_go" "$normalized_go"
  generate_lattigo_runner "$runner_go" "$tag" "$arity" "$input_go" "$params_output"

  if [[ "$params_only_mode" == "1" ]]; then
    print_header "$TEST_NAME LATTIGO $tag" "Extracting backend params (skipping encrypted computation)"
  else
    print_header "$TEST_NAME LATTIGO $tag" "Running encrypted computation + plaintext check"
  fi

  local go_cmd="cd \"$REPO_ROOT\" && go run \"$normalized_go\" \"$runner_go\""
  if [[ "$params_only_mode" == "1" ]]; then
    go_cmd="cd \"$REPO_ROOT\" && STOP_AFTER_BACKEND_PARAM_EXTRACTION=1 go run \"$normalized_go\" \"$runner_go\""
  fi

  if ! run_command bash -c "$go_cmd"; then
    run_failed=1
    record_runtime_failure "$TEST_NAME" "lattigo-${tag}"
    if [[ "$NON_FATAL_RUNTIME_CHECKS" != "1" ]]; then
      return 1
    fi
  fi

  if [[ "$params_only_mode" == "1" && ! -s "$params_output" ]]; then
    echo "WARNING: Missing Lattigo execution params for $TEST_NAME/$tag at $params_output." >&2
    record_runtime_failure "$TEST_NAME" "lattigo-${tag}-missing-execution-params"
    if [[ "$NON_FATAL_RUNTIME_CHECKS" != "1" ]]; then
      return 1
    fi
  fi
}

process_test() {
  TEST_NAME="$1"
  local test_start_epoch
  test_start_epoch="$(date +%s)"
  TEST_DIR="$SCRIPT_DIR/$TEST_NAME"
  INPUT_MLIR="$TEST_DIR/$TEST_NAME.mlir"

  if [[ ! -f "$INPUT_MLIR" ]]; then
    echo "WARNING: Skipping $TEST_NAME (missing input: $INPUT_MLIR)" >&2
    return 0
  fi

  LOG_DIR="$TEST_DIR/logs"
  DATA_DIR="$SCRIPT_DIR/data"
  mkdir -p "$LOG_DIR" "$DATA_DIR"
  MASTER_LOG="$LOG_DIR/execution_${TIMESTAMP}.log"

  MIDDLE_MLIR="$TEST_DIR/$TEST_NAME-middle.mlir"
  GAP_MONO_MLIR="$TEST_DIR/$TEST_NAME-middle-params-gap-mono.mlir"

  print_header "$TEST_NAME" "Pre-BGV secret/arithmetic lowering"
  set_test_state "PREPROCESS"
  run_command bash -c "\"$HEIR_OPT\" \
    --canonicalize \
    --secretize \
    --mlir-to-secret-arithmetic \
    --secret-insert-mgmt-bgv \
    \"$INPUT_MLIR\" > \"$MIDDLE_MLIR\""

  local direct_mlir direct_bgv_mlir
  set_test_state "ANNOTATE_DIRECT"
  direct_mlir="$(annotate_algorithm DIRECT)"
  set_test_state "LOWER_DIRECT"
  direct_bgv_mlir="$(lower_to_bgv "$direct_mlir" direct)"
  set_test_state "GEN_OPENFHE_DIRECT"
  generate_openfhe "$direct_bgv_mlir" direct

  local closed_mlir closed_bgv_mlir
  set_test_state "ANNOTATE_CLOSED"
  closed_mlir="$(annotate_algorithm CLOSED)"
  set_test_state "LOWER_CLOSED"
  closed_bgv_mlir="$(lower_to_bgv "$closed_mlir" closed)"
  set_test_state "GEN_OPENFHE_CLOSED"
  generate_openfhe "$closed_bgv_mlir" closed

  local bisection_mlir bisection_bgv_mlir
  set_test_state "ANNOTATE_BISECTION"
  bisection_mlir="$(annotate_algorithm BISECTION)"
  set_test_state "LOWER_BISECTION"
  bisection_bgv_mlir="$(lower_to_bgv "$bisection_mlir" bisection)"
  set_test_state "GEN_OPENFHE_BISECTION"
  generate_openfhe "$bisection_bgv_mlir" bisection

  set_test_state "POSTPROCESS_OPENFHE_INCLUDES"
  fix_openfhe_include_paths direct closed bisection

  local greedy_mlir greedy_bgv_mlir
  set_test_state "ANNOTATE_GREEDY"
  greedy_mlir="$(annotate_algorithm GREEDY)"
  set_test_state "LOWER_GREEDY"
  greedy_bgv_mlir="$(lower_to_bgv "$greedy_mlir" greedy)"
  set_test_state "GEN_LATTIGO_GREEDY"
  generate_lattigo "$greedy_bgv_mlir" greedy

  local gap_mlir gap_bgv_mlir
  set_test_state "RUN_GAP_MONO"
  run_gap_mono
  set_test_state "LOWER_GAP_MONO"
  gap_bgv_mlir="$(lower_to_bgv "$GAP_MONO_MLIR" gap-mono)"
  set_test_state "GEN_LATTIGO_GAP_MONO"
  generate_lattigo "$gap_bgv_mlir" gap-mono

  set_test_state "RUN_OPENFHE_CHECKS"
  run_openfhe_checks

  local lattigo_tags
  read -r -a lattigo_tags <<< "$LATTIGO_APPROACHES"
  for tag in "${lattigo_tags[@]}"; do
    set_test_state "RUN_LATTIGO_${tag^^}"
    run_lattigo_check "$tag"
  done

  # set_test_state "EXTRACT_OPERATION_COUNTS"
  # extract_operation_counts_for_test

  if [[ "$STOP_AFTER_PARAM_EXTRACTION" == "1" ]]; then
    set_test_state "DONE_BACKEND_PARAMS_ONLY"
    print_header "$TEST_NAME DONE" "Stopped after backend parameter extraction"
    echo "  $MIDDLE_MLIR"
    echo "  $direct_bgv_mlir"
    echo "  $closed_bgv_mlir"
    echo "  $bisection_bgv_mlir"
    if [[ -n "$greedy_bgv_mlir" ]]; then
      echo "  $greedy_bgv_mlir"
    fi
    if [[ -n "$gap_bgv_mlir" ]]; then
      echo "  $gap_bgv_mlir"
    fi
    local test_elapsed
    test_elapsed=$(( $(date +%s) - test_start_epoch ))
    echo "Elapsed: $(format_duration "$test_elapsed") (${test_elapsed}s)"
    return 0
  fi

  set_test_state "DONE"
  print_header "$TEST_NAME DONE" "Generated outputs"
  echo "  $MIDDLE_MLIR"
  echo "  $direct_bgv_mlir"
  echo "  $closed_bgv_mlir"
  echo "  $bisection_bgv_mlir"
  if [[ -n "$greedy_bgv_mlir" ]]; then
    echo "  $greedy_bgv_mlir"
  fi
  if [[ -n "$gap_bgv_mlir" ]]; then
    echo "  $gap_bgv_mlir"
  fi
  local test_elapsed
  test_elapsed=$(( $(date +%s) - test_start_epoch ))
  echo "Elapsed: $(format_duration "$test_elapsed") (${test_elapsed}s)"
}

declare -A TEST_STATE_MAP=()
declare -A TEST_WORKER_LOG_MAP=()
declare -A TEST_RUNTIME_COUNT_MAP=()
declare -A TEST_STATUS_FILE_MAP=()
declare -A TEST_START_EPOCH_MAP=()
declare -A TEST_DURATION_MAP=()
declare -A FAMILY_MAX_LEVEL_MAP=()

set_test_state() {
  local state="$1"
  if [[ -n "${TEST_STATUS_FILE:-}" ]]; then
    printf "%s\n" "$state" > "$TEST_STATUS_FILE"
  fi
}

render_status_table() {
  local completed="$1"
  local total="$2"
  local running="$3"
  local passed="$4"
  local failed="$5"
  local runtime_failed="$6"
  local started="$7"
  local parallel_start="$8"
  local run_root="$9"
  local elapsed=$(( $(date +%s) - parallel_start ))

  printf '\033[H\033[2J'
  echo "HEIR parallel test dashboard"
  echo "Run root: ${run_root}"
  echo "Elapsed: $(format_duration "$elapsed") (${elapsed}s)"
  echo "State legend: PENDING, RUNNING, PASSED, PASSED_RTF, FAILED"
  echo "Started: ${started}/${total} | Completed: ${completed}/${total} | Running: ${running} | Passed: ${passed} | Failed: ${failed} | Runtime-fail entries: ${runtime_failed}"
  echo
  printf "| %-24s | %-12s | %-10s | %-8s | %-32s |\n" "Test" "State" "Done In" "RTF" "Worker log"
  printf "|-%-24s-|-%-12s-|-%-10s-|-%-8s-|-%-32s-|\n" "------------------------" "------------" "----------" "--------" "--------------------------------"

  local test_name
  for test_name in "${TESTS[@]}"; do
    local state="${TEST_STATE_MAP[$test_name]:-PENDING}"
    local status_file="${TEST_STATUS_FILE_MAP[$test_name]:-}"
    if [[ -n "$status_file" && -f "$status_file" && "$state" == "RUNNING" ]]; then
      local live_state
      live_state="$(head -n 1 "$status_file")"
      if [[ -n "$live_state" ]]; then
        state="$live_state"
      fi
    fi
    local runtime_count="${TEST_RUNTIME_COUNT_MAP[$test_name]:-0}"
    local done_in="-"
    if [[ "$state" == PASSED* ]]; then
      local duration_seconds="${TEST_DURATION_MAP[$test_name]:-}"
      if [[ -n "$duration_seconds" ]]; then
        done_in="$(format_duration "$duration_seconds")"
      fi
    fi
    local worker_log="${TEST_WORKER_LOG_MAP[$test_name]:--}"
    local worker_name
    worker_name="$(basename "$worker_log")"
    printf "| %-24s | %-12s | %-10s | %-8s | %-32s |\n" \
      "$test_name" "$state" "$done_in" "$runtime_count" "$worker_name"
  done
}

run_parallel_tests() {
  local total="${#TESTS[@]}"
  local parallel_start
  parallel_start="$(date +%s)"
  local run_root
  run_root="$(mktemp -d "${SCRIPT_DIR}/.parallel_${TEST_GROUP}_${TIMESTAMP}_XXXX")"

  local -a queue
  queue=("${TESTS[@]}")

  local -a running_tests=()
  local -a running_pids=()
  local started=0
  local completed=0
  local passed=0
  local failed=0
  local runtime_failed=0

  echo "Running ${total} tests with PARALLEL_JOBS=${PARALLEL_JOBS}"
  echo "Per-test worker logs: ${run_root}"

  local test_name
  for test_name in "${TESTS[@]}"; do
    TEST_STATE_MAP["$test_name"]="PENDING"
    TEST_WORKER_LOG_MAP["$test_name"]="-"
    TEST_RUNTIME_COUNT_MAP["$test_name"]=0
    TEST_STATUS_FILE_MAP["$test_name"]="${run_root}/${test_name}.status"
    TEST_START_EPOCH_MAP["$test_name"]=""
    TEST_DURATION_MAP["$test_name"]=""
  done

  render_status_table "$completed" "$total" 0 "$passed" "$failed" "$runtime_failed" "$started" "$parallel_start" "$run_root"

  while [[ "$completed" -lt "$total" ]]; do
    while [[ "${#running_pids[@]}" -lt "$PARALLEL_JOBS" && "$started" -lt "$total" ]]; do
      test_name="${queue[$started]}"
      local worker_log="${run_root}/${test_name}.worker.log"
      local runtime_fail_file="${run_root}/${test_name}.runtime_failures"
      local status_file="${run_root}/${test_name}.status"

      (
        FAILED_RUNTIME_CHECKS=()
        TEST_STATUS_FILE="$status_file"
        set_test_state "RUNNING"
        if process_test "$test_name"; then
          rc=0
        else
          rc=$?
        fi

        if [[ ${#FAILED_RUNTIME_CHECKS[@]} -gt 0 ]]; then
          printf "%s\n" "${FAILED_RUNTIME_CHECKS[@]}" > "$runtime_fail_file"
        fi

        exit "$rc"
      ) >"$worker_log" 2>&1 &

      running_tests+=("$test_name")
      running_pids+=("$!")
      TEST_STATE_MAP["$test_name"]="RUNNING"
      TEST_WORKER_LOG_MAP["$test_name"]="$worker_log"
      TEST_STATUS_FILE_MAP["$test_name"]="$status_file"
      TEST_START_EPOCH_MAP["$test_name"]="$(date +%s)"
      started=$((started + 1))
    done

    ACTIVE_PARALLEL_PIDS=("${running_pids[@]}")

    local -a next_running_tests=()
    local -a next_running_pids=()

    local i
    for i in "${!running_pids[@]}"; do
      local pid="${running_pids[$i]}"
      test_name="${running_tests[$i]}"
      local worker_log="${run_root}/${test_name}.worker.log"
      local runtime_fail_file="${run_root}/${test_name}.runtime_failures"

      if kill -0 "$pid" 2>/dev/null; then
        next_running_tests+=("$test_name")
        next_running_pids+=("$pid")
      else
        local rc=0
        if wait "$pid"; then
          rc=0
        else
          rc=$?
        fi

        completed=$((completed + 1))
        local end_epoch
        end_epoch="$(date +%s)"
        local start_epoch="${TEST_START_EPOCH_MAP[$test_name]:-}"
        if [[ -n "$start_epoch" ]]; then
          TEST_DURATION_MAP["$test_name"]="$((end_epoch - start_epoch))"
        fi

        local test_runtime_fail_count=0
        if [[ -f "$runtime_fail_file" ]]; then
          while IFS= read -r failure; do
            [[ -n "$failure" ]] || continue
            FAILED_RUNTIME_CHECKS+=("$failure")
            test_runtime_fail_count=$((test_runtime_fail_count + 1))
            runtime_failed=$((runtime_failed + 1))
          done < "$runtime_fail_file"
        fi
        TEST_RUNTIME_COUNT_MAP["$test_name"]="$test_runtime_fail_count"

        if [[ "$rc" -eq 0 ]]; then
          passed=$((passed + 1))
          if [[ "$test_runtime_fail_count" -gt 0 ]]; then
            TEST_STATE_MAP["$test_name"]="PASSED_RTF"
          else
            TEST_STATE_MAP["$test_name"]="PASSED"
          fi
        else
          failed=$((failed + 1))
          FAILED_TESTS+=("$test_name")
          TEST_STATE_MAP["$test_name"]="FAILED"
        fi
      fi
    done

    running_tests=("${next_running_tests[@]}")
    running_pids=("${next_running_pids[@]}")
    ACTIVE_PARALLEL_PIDS=("${running_pids[@]}")

    render_status_table "$completed" "$total" "${#running_pids[@]}" "$passed" "$failed" "$runtime_failed" "$started" "$parallel_start" "$run_root"
    sleep "$DASHBOARD_REFRESH_SEC"
  done

  ACTIVE_PARALLEL_PIDS=()

  render_status_table "$completed" "$total" 0 "$passed" "$failed" "$runtime_failed" "$started" "$parallel_start" "$run_root"
  echo
  local parallel_elapsed
  parallel_elapsed=$(( $(date +%s) - parallel_start ))
  echo "Parallel run elapsed: $(format_duration "$parallel_elapsed") (${parallel_elapsed}s)"
}

discover_tests "$@"

if [[ ${#TESTS[@]} -eq 0 ]]; then
  echo "ERROR: No tests selected. Use TEST_GROUP/TEST_NAMES or positional test names." >&2
  exit 1
fi

echo "Selected tests: ${TESTS[*]}"

if [[ "$LIST_TESTS_ONLY" == "1" ]]; then
  exit 0
fi

build_tools

run_parallel_tests

if [[ ${#FAILED_TESTS[@]} -gt 0 ]]; then
  echo
  echo "Compile/translation failures:" >&2
  for failure in "${FAILED_TESTS[@]}"; do
    echo "  $failure" >&2
  done
  exit 1
fi

if [[ ${#FAILED_RUNTIME_CHECKS[@]} -gt 0 ]]; then
  echo
  echo "Runtime check failures:" >&2
  for failure in "${FAILED_RUNTIME_CHECKS[@]}"; do
    echo "  $failure" >&2
  done

  if [[ "$NON_FATAL_RUNTIME_CHECKS" == "1" ]]; then
    echo "WARNING: Completed with runtime check failures (NON_FATAL_RUNTIME_CHECKS=1)." >&2
    exit 0
  fi

  exit 1
fi

echo
print_header "DONE" "All selected tests completed successfully"
total_elapsed=$(( $(date +%s) - SCRIPT_START_EPOCH ))
echo "Total script elapsed: $(format_duration "$total_elapsed") (${total_elapsed}s)"
