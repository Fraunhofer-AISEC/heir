# See README.md for setup required to run these tests

load("@heir//tests/Examples/lattigo:test.bzl", "heir_lattigo_lib")
load("@rules_go//go:def.bzl", "go_test")

package(default_applicable_licenses = ["@heir//:license"])

# This is a Google-internal hack to avoid issues with automated tooling that
# expects very specific go package layout with respect to build targets.
# @unused
glaze_ignore = [
    "bfv_debug.go",
]

heir_lattigo_lib(
    name = "mult_indep_16_debug",
    extra_srcs = ["bfv_debug.go"],
    go_library_name = "main",
    heir_opt_flags = [
        "--annotate-module=backend=lattigo scheme=bfv",
        "--mlir-to-bfv=noise-model=bfv-noise-bmcm23 \
          annotate-noise-bound=true",
        "--scheme-to-lattigo=insert-debug-handler-calls=true",
    ],
    mlir_src = "@heir//tests/Examples/common:mult_indep_16.mlir",
)

# For Google-internal reasons we must separate the go_test rules from the macro
# above.

# TODO(#1614): re-enable as non-manual
go_test(
    name = "multindep16debug_test",
    srcs = ["mult_indep_16_debug_test.go"],
    embed = [":main"],
    tags = [
        "manual",
        "notap",
    ],
)
