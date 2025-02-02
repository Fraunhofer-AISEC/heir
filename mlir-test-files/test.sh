name="dot8"

# TODO Fix with new pipeline
bazel run //tools:heir-opt -- --secretize --mlir-to-secret-arithmetic --secret-insert-mgmt-bgv $PWD/$name/$name.mlir > $PWD/$name/$name-middle.mlir
bazel run //tools:heir-opt -- --annotate-parameters="plaintext-modulus=65537 ring-dimension=8192" $PWD/$name/$name-middle.mlir > $PWD/$name/$name-middle-params.mlir
bazel run //tools:heir-opt -- --mlir-to-bgv="ciphertext-degree=8 insert-mgmt=false" -scheme-to-openfhe="entry-function=func" $PWD/$name/$name-middle-params.mlir > $PWD/$name/$name-openfhe.mlir

bazel run //tools:heir-translate -- --emit-openfhe-pke-header "$PWD/$name/${name}-openfhe.mlir" > "$name/$name.h"
bazel run //tools:heir-translate -- --emit-openfhe-pke "$PWD/$name/${name}-openfhe.mlir" > "$name/$name.cpp"

# Replace the include path in the generated C++ files
sed -i 's|#include "openfhe/pke/openfhe.h"|#include "src/pke/include/openfhe.h" // from @openfhe|g' "$name/$name.h" "$name/$name.cpp"

bazel run //mlir-test-files:main_dot8
