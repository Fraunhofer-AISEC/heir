#!/usr/bin/env bash
set -euo pipefail

# Usage: ./build_and_run.sh <NAME>
# Expects:
#   - MLIR at: <NAME>/<NAME>.mlir
#   - Main at: <NAME>/<NAME>_main.rs
NAME="${1:-foo}"
MLIR_SRC="${NAME}/${NAME}.mlir"
MAIN_SRC="${NAME}/${NAME}_main.rs"

# Subfolder where all work happens (per-name)
PROJECT_DIR="${NAME}/tfhe/heir_tfhe_project"
OUT_DIR="${PROJECT_DIR}/generated"
SRC_DIR="${PROJECT_DIR}/src"
GENERATED_RS="${NAME}_lib.rs"
CARGO_PACKAGE="heir_tfhe_project"
BIN_NAME="${NAME}"

command -v bazel >/dev/null 2>&1 || { echo "Error: bazel not found"; exit 1; }
command -v cargo >/dev/null 2>&1 || { echo "Error: cargo not found"; exit 1; }

# Validate inputs
if [ ! -f "$MLIR_SRC" ]; then
  echo "Error: MLIR source '$MLIR_SRC' not found. Expected at <NAME>/<NAME>.mlir"
  exit 1
fi

if [ ! -f "$MAIN_SRC" ]; then
  echo "Error: MAIN source '$MAIN_SRC' not found. Expected at <NAME>/<NAME>_main.rs"
  exit 1
fi

# Prepare subfolder structure
mkdir -p "$OUT_DIR" "$SRC_DIR"

# 1) heir-opt via Bazel -> write to subfolder
HEIR_OPT_OUT="${OUT_DIR}/$(basename "${MLIR_SRC%.*}").heir_opt.mlir"
echo "Running heir-opt..."
bazel run //tools:heir-opt -- --mlir-to-cggi --scheme-to-tfhe-rs "$PWD/${MLIR_SRC}" > "${HEIR_OPT_OUT}"

# 2) heir-translate via Bazel -> write to subfolder
GENERATED_RS_PATH="${OUT_DIR}/${GENERATED_RS}"
echo "Running heir-translate..."
bazel run //tools:heir-translate -- --emit-tfhe-rust "$PWD/${HEIR_OPT_OUT}" > "${GENERATED_RS_PATH}"

# 3) Create Cargo project in subfolder
if [ ! -f "${PROJECT_DIR}/Cargo.toml" ]; then
  echo "Creating Cargo project in '${PROJECT_DIR}'..."
  cargo init --name "$CARGO_PACKAGE" --bin "$PROJECT_DIR" >/dev/null
  rm -f "${PROJECT_DIR}/src/main.rs"
fi

# 4) Write Cargo.toml with required dependencies/features
cat > "${PROJECT_DIR}/Cargo.toml" <<EOF
[package]
name = "heir_tfhe_project"
version = "0.1.0"
edition = "2021"

[dependencies]
serde = { version = "1", features = ["derive"] }
rayon = "1"
tfhe = { version = "*", features = ["boolean", "shortint", "integer", "seeder_unix"] }

[[bin]]
name = "${BIN_NAME}"
path = "src/main_${BIN_NAME}.rs"
EOF

# 5) lib.rs includes generated code from subfolder
cat > "${SRC_DIR}/lib.rs" <<EOF
pub mod generated {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/generated/${GENERATED_RS}"));
}
EOF

# 6) Copy the provided main into subfolder, renaming to match the bin target
cp "$MAIN_SRC" "${SRC_DIR}/main_${BIN_NAME}.rs"
cp bench/bench.rs "${SRC_DIR}/"

# 7) Build and run from subfolder
echo "Building and running in '${PROJECT_DIR}'..."
cargo run --release --manifest-path "${PROJECT_DIR}/Cargo.toml" --bin "${BIN_NAME}"