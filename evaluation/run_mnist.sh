#!/bin/bash

timestamp=$(date +"%Y%m%d_%H%M%S")
filename=$(basename "$1")
target_dir="$PWD/log/ir_${filename}_${timestamp}"

mkdir -p "$target_dir"

bazel run //tools:heir-opt -- "$PWD/$1" --pass-pipeline='builtin.module(
  wrap-generic,
  select-rewrite,
  compare-to-sign-rewrite,
  full-loop-unroll,
  apply-folders,
  canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},
  insert-rotate,
  cse,
  canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},
  cse,
  collapse-insertion-chains,
  sccp,
  canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},
  cse,
  rotate-and-reduce,
  sccp,
  apply-folders,
  canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},
  cse,
  polynomial-approximation,
  lower-polynomial-eval{method=auto},
  canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},
  cse,
  new-layout-propagation{ciphertext-size=1024},
  fold-convert-layout-into-assign-layout,
  layout-optimization{ciphertext-size=1024},
  convert-to-ciphertext-semantics{ciphertext-size=1024},
  implement-rotate-and-reduce,
  operation-balancer,
  func.func(
    split-preprocessing
  ),
  tensor-linalg-to-affine-loops,
  func.func(
    affine-expand-index-ops
  ),
  func.func(
    affine-simplify-structures
  ),
  func.func(
    affine-loop-normalize{promote-single-iter=true}
  ),
  canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},
  cse,
  add-client-interface{ciphertext-size=1024 enable-layout-assignment=true}
)' --mlir-print-ir-after-all --mlir-print-ir-tree-dir="$target_dir" > "$target_dir/${filename}.secretized_arith.mlir"

# bazel run //tools:heir-opt -- --generate-param-bgv="model=bgv-noise-mono" "$target_dir/$1_secret_arith.mlir" > "$target_dir/$1_secret_arith_param.mlir"

bazel run //tools:heir-opt -- --mlir-to-bgv="ciphertext-degree=1024 noise-model=bgv-noise-mono" --scheme-to-openfhe="entry-function=main" --debug --mlir-print-ir-after-all --mlir-print-ir-tree-dir="$target_dir/mlir-to-bgv" "$target_dir/${filename}.secretized_arith.mlir" > "$target_dir/${filename}.bgv.mlir"