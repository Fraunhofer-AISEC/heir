#!/bin/bash
set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 <filename.mlir>"
    exit 1
fi

IN="$1"
FILENAME=$(basename "$IN" .mlir)
OUTDIR="$PWD/${FILENAME}/openfhe"

mkdir -p "$OUTDIR"

# Convert to BGV and OpenFHE MLIR
bazel run //tools:heir-opt -- "$PWD/${FILENAME}/${FILENAME}.mlir" --mlir-to-bgv='ciphertext-degree=8' --scheme-to-openfhe='entry-function=foo' > "$OUTDIR/${FILENAME}-openfhe.mlir"

# Emit OpenFHE headers and sources into openfhe dir
bazel run //tools:heir-translate -- --emit-openfhe-pke-header --openfhe-include-type=source-relative "$OUTDIR/${FILENAME}-openfhe.mlir" > "$OUTDIR/${FILENAME}.h"

bazel run //tools:heir-translate -- --emit-openfhe-pke --openfhe-include-type=source-relative "$OUTDIR/${FILENAME}-openfhe.mlir" > "$OUTDIR/${FILENAME}.cpp"

# Copy benchmark main file
cp "${FILENAME}/${FILENAME}_main.cpp" "$OUTDIR/${FILENAME}_main.cpp"

# BUILD file inside openfhe package
cat > "$OUTDIR/BUILD" << EOL
cc_library(
    name = "${FILENAME}_codegen",
    srcs = ["${FILENAME}.cpp"],
    hdrs = ["${FILENAME}.h"],
    deps = ["@openfhe//:pke"],
)

cc_binary(
    name = "${FILENAME}_main",
    srcs = ["${FILENAME}_main.cpp"],
    deps = [
        ":${FILENAME}_codegen",
        "//scheme-test/bench",
        "@openfhe//:pke",
        "@openfhe//:core",
    ],
)
EOL

# Run the binary from the openfhe package
bazel run "//scheme-test/${FILENAME}/openfhe:${FILENAME}_main"

# Optional cleanup
rm -f "$OUTDIR/BUILD"