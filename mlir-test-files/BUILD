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

cc_library(
    name = "dot8_common",
    srcs = ["dot8/dot8_common.cpp"],
    hdrs = ["dot8/dot8_common.h"],
    deps = [
        ":params",
        "@openfhe//:pke",
    ],
)

cc_binary(
    name = "main_dot8_bisection",
    srcs = ["dot8/main_dot8_bisection.cpp"],
    deps = [
        ":dot8_common",
        ":dot8_bisection",
    ],
)

cc_binary(
    name = "main_dot8_openfhe",
    srcs = ["dot8/main_dot8_openfhe.cpp"],
    deps = [
        ":dot8_common",
        ":dot8_openfhe",
    ],
)

cc_binary(
    name = "main_dot8_direct",
    srcs = ["dot8/main_dot8_direct.cpp"],
    deps = [
        ":dot8_common",
        ":dot8_direct",
    ],
)

cc_library(
    name = "dot8_bisection",
    srcs = ["dot8/dot8_bisection.cpp"],
    hdrs = ["dot8/dot8_bisection.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "dot8_openfhe",
    srcs = ["dot8/dot8_openfhe.cpp"],
    hdrs = ["dot8/dot8_openfhe.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "dot8_direct",
    srcs = ["dot8/dot8_direct.cpp"],
    hdrs = ["dot8/dot8_direct.h"],
    deps = [
        "@openfhe//:pke",
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

cc_library(
    name = "psi_common",
    srcs = ["psi/psi_common.cpp"],
    hdrs = ["psi/psi_common.h"],
    deps = [
        ":params",
        "@openfhe//:pke",
    ],
)

cc_binary(
    name = "main_psi_bisection",
    srcs = ["psi/main_psi_bisection.cpp"],
    deps = [
        ":psi_common",
        ":psi_bisection",
    ],
)

cc_binary(
    name = "main_psi_openfhe",
    srcs = ["psi/main_psi_openfhe.cpp"],
    deps = [
        ":psi_common",
        ":psi_openfhe",
    ],
)

cc_library(
    name = "psi_bisection",
    srcs = ["psi/psi_bisection.cpp"],
    hdrs = ["psi/psi_bisection.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "psi_openfhe",
    srcs = ["psi/psi_openfhe.cpp"],
    hdrs = ["psi/psi_openfhe.h"],
    deps = [
        "@openfhe//:pke",
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

cc_binary(
    name = "main_psi_direct",
    srcs = [
        "psi/psi_direct.cpp",
        "psi/psi_direct.h",
    ],
    deps = [
        "@openfhe//:openfhe",
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

cc_library(
    name = "mult5_common",
    srcs = ["mult5/mult5_common.cpp"],
    hdrs = ["mult5/mult5_common.h"],
    deps = [
        ":params",
        "@openfhe//:pke",
    ],
)

cc_binary(
    name = "main_mult5_bisection",
    srcs = ["mult5/main_mult5_bisection.cpp"],
    deps = [
        ":mult5_common",
        ":mult5_bisection",
    ],
)

cc_binary(
    name = "main_mult5_openfhe",
    srcs = ["mult5/main_mult5_openfhe.cpp"],
    deps = [
        ":mult5_common",
        ":mult5_openfhe",
    ],
)

cc_library(
    name = "mult5_bisection",
    srcs = ["mult5/mult5_bisection.cpp"],
    hdrs = ["mult5/mult5_bisection.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "mult5_openfhe",
    srcs = ["mult5/mult5_openfhe.cpp"],
    hdrs = ["mult5/mult5_openfhe.h"],
    deps = [
        "@openfhe//:pke",
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

cc_binary(
    name = "main_mult5_direct",
    srcs = [
        "mult5/mult5_direct.cpp",
        "mult5/mult5_direct.h",
    ],
    deps = [
        "@openfhe//:openfhe",
    ],
)

cc_library(
    name = "add-256-0",
    srcs = ["add-256-0/add-256-0.cpp"],
    hdrs = ["add-256-0/add-256-0.h"],
    deps = [
        "@openfhe//:pke"
    ],
)

cc_binary(
    name = "main_add-256-0",
    srcs = ["add-256-0/main_add-256-0.cpp"],
    deps = [
        ":add-256-0",
        ":params",
        "@openfhe//:core",
        "@openfhe//:pke",
    ],
)

cc_binary(
    name = "main_add-256-0_direct",
    srcs = [
        "add-256-0/add-256-0_direct.cpp",
        "add-256-0/add-256-0_direct.h",
    ],
    deps = [
        "@openfhe//:openfhe",
    ],
)

cc_library(
    name = "add-64-3",
    srcs = ["add-64-3/add-64-3.cpp"],
    hdrs = ["add-64-3/add-64-3.h"],
    deps = [
        "@openfhe//:pke"
    ],
)

cc_binary(
    name = "main_add-64-3",
    srcs = ["add-64-3/main_add-64-3.cpp"],
    deps = [
        ":add-64-3",
        ":params",
        "@openfhe//:core",
        "@openfhe//:pke",
    ],
)

cc_binary(
    name = "main_add-64-3_direct",
    srcs = [
        "add-64-3/add-64-3_direct.cpp",
        "add-64-3/add-64-3_direct.h",
    ],
    deps = [
        "@openfhe//:openfhe",
    ],
)