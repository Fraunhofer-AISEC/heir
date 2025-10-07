#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: $0 <filename>"
    exit 1
fi

# Get the filename without extension
FILENAME=$(basename "$1" .mlir)

bazel run //tools:heir-opt -- "$PWD/$1" --mlir-to-bgv='ciphertext-degree=8' --scheme-to-openfhe='entry-function=foo' > "$PWD/${FILENAME}-openfhe.mlir"

bazel run //tools:heir-translate -- --emit-openfhe-pke-header --openfhe-include-type=source-relative "$PWD/${FILENAME}-openfhe.mlir" > "${FILENAME}.h"

bazel run //tools:heir-translate -- --emit-openfhe-pke --openfhe-include-type=source-relative "$PWD/${FILENAME}-openfhe.mlir" > "${FILENAME}.cpp"

cat > BUILD << EOL
cc_library(
    name = "${FILENAME}_codegen",
    srcs = ["${FILENAME}.cpp"],
    hdrs = ["${FILENAME}.h"],
    deps = ["@openfhe//:pke"],
)

# An executable build target that contains your main function and links
# against the above.
cc_binary(
    name = "${FILENAME}_main",
    srcs = ["${FILENAME}_main.cpp", "bench.hpp"],
    deps = [
        ":${FILENAME}_codegen",
        "@openfhe//:pke",
        "@openfhe//:core",
    ],
)
EOL

bazel run "${FILENAME}_main"

rm -rf BUILD


