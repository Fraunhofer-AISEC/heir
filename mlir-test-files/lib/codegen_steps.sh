#!/usr/bin/env bash

# Generate OpenFHE artifacts from BGV MLIR.
generate_openfhe() {
  local bgv_mlir="$1"
  local tag="$2"
  local openfhe_mlir="$TEST_DIR/$TEST_NAME-openfhe-$tag.mlir"
  local header="$TEST_DIR/${TEST_NAME}_${tag}.h"
  local cpp="$TEST_DIR/${TEST_NAME}_${tag}.cpp"

  print_header "$TEST_NAME $tag" "Lowering BGV to OpenFHE"
  run_command bash -c '"$1" \
    --canonicalize \
    --bgv-to-lwe \
    --ckks-to-lwe \
    --lwe-to-openfhe \
    --canonicalize \
    --cse \
    "$2" \
    --openfhe-fast-rotation-precompute \
    --boolean-vectorize \
    --canonicalize \
    --cse \
    --openfhe-alloc-to-inplace \
    --remove-unused-pure-call \
    --remove-dead-values \
    --cse \
    --canonicalize \
    --symbol-dce \
    "$3" > "$4"' _ \
    "$HEIR_OPT" \
    "--openfhe-configure-crypto-context=entry-function=func"\
    "$bgv_mlir" \
    "$openfhe_mlir"

  print_header "$TEST_NAME $tag" "Generating OpenFHE header and implementation"
  run_command bash -c '"$1" "$2" "$3" > "$4"' _ \
    "$HEIR_TRANSLATE" \
    --emit-openfhe-pke-header \
    "$openfhe_mlir" \
    "$header"
  run_command bash -c '"$1" "$2" "$3" > "$4"' _ \
    "$HEIR_TRANSLATE" \
    --emit-openfhe-pke \
    "$openfhe_mlir" \
    "$cpp"
}

# Generate Lattigo artifacts from BGV MLIR.
generate_lattigo() {
  local bgv_mlir="$1"
  local tag="$2"
  local lattigo_mlir="$TEST_DIR/$TEST_NAME-lattigo-$tag.mlir"
  local go_out="$TEST_DIR/${TEST_NAME}_${tag}_lattigo.go"

  print_header "$TEST_NAME $tag" "Lowering BGV to Lattigo"
  run_command bash -c '"$1" "$2" "$3" > "$4"' _ \
    "$HEIR_OPT" \
    --scheme-to-lattigo=entry-function=func \
    "$bgv_mlir" \
    "$lattigo_mlir"

  print_header "$TEST_NAME $tag" "Generating Lattigo Go implementation"
  run_command bash -c '"$1" "$2" "$3" > "$4"' _ \
    "$HEIR_TRANSLATE" \
    --emit-lattigo \
    "$lattigo_mlir" \
    "$go_out"
}

normalize_lattigo_file() {
  local input_go="$1"
  local output_go="$2"

  run_command bash -c "sed -e '1s/^package[[:space:]]*$/package main/' -e 's/^func func(/func fheFunc(/' \"$input_go\" > \"$output_go\""
}

lattigo_expected_value_for_test() {
  case "$1" in
    add-eq-1)
      echo 2
      ;;
    add-eq-6)
      echo -23492
      ;;
    add-eq-10)
      echo 29445
      ;;
    add-eq-4)
      echo 677
      ;;
    add-eq-2)
      echo 5
      ;;
    form-decreasing)
      echo -424
      ;;
    form-*)
      echo 677
      ;;
    add-num-32)
      echo 677
      ;;
    add-num-64)
      echo 677
      ;;
    add-num-128)
      echo 677
      ;;
    *)
      echo 1
      ;;
  esac
}

# Render the Lattigo runner from a reusable template.
generate_lattigo_runner() {
  local runner_go="$1"
  local approach="$2"
  local arity="$3"
  local input_go="$4"
  local params_output="$5"
  local expected_value
  expected_value="$(lattigo_expected_value_for_test "$TEST_NAME")"

  run_command python3 "$SCRIPT_DIR/render_lattigo_runner.py" \
    "$SCRIPT_DIR/lattigo_runner_template.go.tmpl" \
    "$runner_go" \
    "$TEST_NAME" \
    "$approach" \
    "$arity" \
    "$input_go" \
    "$expected_value" \
    "$params_output"
}
