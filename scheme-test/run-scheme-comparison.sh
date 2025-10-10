#!/bin/bash
set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 <filename.mlir>"
    exit 1
fi

IN="$1"
FILENAME=$(basename "$IN" .mlir)

# Annotate scheme info (debug; no output file)
bazel run //tools:heir-opt -- "$PWD/${FILENAME}/${FILENAME}.mlir" --annotate-scheme-info --debug

bash ./run-openfhe.sh "${FILENAME}"
bash ./run-tfhe-rust-hl.sh "${FILENAME}"
bash ./run-tfhe-rust.sh "${FILENAME}"

