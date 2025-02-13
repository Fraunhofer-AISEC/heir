# CC library for the params helpers
cc_library(
    name = "params",
    srcs = ["params.cpp"],
    hdrs = ["params.h"],
    deps = [
        "@openfhe//:core",
        "@openfhe//:pke",
    ],
)


cc_library(
    name = "dot8",
    srcs = ["dot8/dot8.cpp"],
    hdrs = ["dot8/dot8.h"],
    deps = [
        "@openfhe//:pke"
    ],
)

# An executable build target that contains your main function and links
# against the above.
cc_binary(
    name = "main_dot8",
    srcs = ["dot8/main_dot8.cpp"],
    deps = [
        ":dot8",
        ":params",
        "@openfhe//:core",
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "psi",
    srcs = ["psi/psi.cpp"],
    hdrs = ["psi/psi.h"],
    deps = [
        "@openfhe//:pke"
    ],
)

cc_binary(
    name = "main_psi",
    srcs = ["psi/main_psi.cpp"],
    deps = [
        ":psi",
        ":params",
        "@openfhe//:core",
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "mult5",
    srcs = ["mult5/mult5.cpp"],
    hdrs = ["mult5/mult5.h"],
    deps = [
        "@openfhe//:pke"
    ],
)

cc_binary(
    name = "main_mult5",
    srcs = ["mult5/main_mult5.cpp"],
    deps = [
        ":mult5",
        ":params",
        "@openfhe//:core",
        "@openfhe//:pke",
    ],
)