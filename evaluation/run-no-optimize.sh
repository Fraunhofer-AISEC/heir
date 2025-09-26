#!/bin/bash

set -e

timestamp=$(date +"%Y%m%d_%H%M%S")
filename=$(basename "$1")
target_dir="$PWD/log/ir_${filename}_${timestamp}"

mkdir -p "$target_dir"

bazel run //tools:heir-opt -- "$PWD/$1" --pass-pipeline='builtin.module(
  secretize,
  wrap-generic,
  select-rewrite,
  compare-to-sign-rewrite,
  full-loop-unroll,
  canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},
  new-layout-propagation{ciphertext-size=1024},
  fold-convert-layout-into-assign-layout,
  canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},
  add-client-interface{ciphertext-size=1024 enable-layout-assignment=true},
  secret-insert-mgmt-bgv
)' --mlir-print-ir-after-all --mlir-print-ir-tree-dir="$target_dir" > "$target_dir/${filename}.secretized_arith.mlir"

bazel run //tools:heir-opt -- --generate-param-bgv="model=bgv-noise-mono" "$target_dir/${filename}.secretized_arith.mlir" > "$target_dir/${filename}_secret_arith_param.mlir"

bazel run //tools:heir-opt -- --mlir-to-bgv="ciphertext-degree=1024 noise-model=bgv-noise-mono" --scheme-to-openfhe="entry-function=main" --debug --mlir-print-ir-after-all --mlir-print-ir-tree-dir="$target_dir/mlir-to-bgv" "$target_dir/${filename}.secretized_arith.mlir" > "$target_dir/${filename}.bgv.mlir"