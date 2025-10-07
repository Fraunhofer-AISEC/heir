#!/bin/bash

# Check if filename argument is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <filename>"
    exit 1
fi

# Get the filename without extension
FILENAME=$(basename "$1" .mlir)

bazel run @heir//tools:heir-opt -- --mlir-to-cggi="data-type=Integer" "$PWD/$1" --mlir-print-ir-after-all --mlir-print-ir-tree-dir="$PWD/log" > "$PWD/${FILENAME}-cggi.mlir"

bazel run @heir//tools:heir-opt -- --scheme-to-tfhe-rs "$PWD/${FILENAME}-cggi.mlir" > "$PWD/${FILENAME}-tfhe.mlir"

bazel run @heir//tools:heir-translate -- --emit-tfhe-rust-hl "$PWD/${FILENAME}-tfhe.mlir"

