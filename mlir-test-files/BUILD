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
    name = "main_add-64-0_common",
    srcs = ["add-64-0/main_add-64-0_common.cpp"],
    hdrs = ["add-64-0/main_add-64-0_common.h"],
    deps = [
        ":params",
        "@openfhe//:pke",
    ],
)

cc_binary(
    name = "main_add-64-0_bisection",
    srcs = ["add-64-0/main_add-64-0_bisection.cpp"],
    deps = [
        ":main_add-64-0_common",
        ":add-64-0_bisection",
    ],
)

cc_binary(
    name = "main_add-64-0_openfhe",
    srcs = ["add-64-0/main_add-64-0_openfhe.cpp"],
    deps = [
        ":main_add-64-0_common",
        ":add-64-0_openfhe",
    ],
)

cc_binary(
    name = "main_add-64-0_direct",
    srcs = ["add-64-0/main_add-64-0_direct.cpp"],
    deps = [
        ":main_add-64-0_common",
        ":add-64-0_direct",
    ],
)

cc_library(
    name = "add-64-0_bisection",
    srcs = ["add-64-0/add-64-0_bisection.cpp"],
    hdrs = ["add-64-0/add-64-0_bisection.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "add-64-0_openfhe",
    srcs = ["add-64-0/add-64-0_openfhe.cpp"],
    hdrs = ["add-64-0/add-64-0_openfhe.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "add-64-0_direct",
    srcs = ["add-64-0/add-64-0_direct.cpp"],
    hdrs = ["add-64-0/add-64-0_direct.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "main_add-64-1_common",
    srcs = ["add-64-1/main_add-64-1_common.cpp"],
    hdrs = ["add-64-1/main_add-64-1_common.h"],
    deps = [
        ":params",
        "@openfhe//:pke",
    ],
)

cc_binary(
    name = "main_add-64-1_bisection",
    srcs = ["add-64-1/main_add-64-1_bisection.cpp"],
    deps = [
        ":main_add-64-1_common",
        ":add-64-1_bisection",
    ],
)

cc_binary(
    name = "main_add-64-1_openfhe",
    srcs = ["add-64-1/main_add-64-1_openfhe.cpp"],
    deps = [
        ":main_add-64-1_common",
        ":add-64-1_openfhe",
    ],
)

cc_binary(
    name = "main_add-64-1_direct",
    srcs = ["add-64-1/main_add-64-1_direct.cpp"],
    deps = [
        ":main_add-64-1_common",
        ":add-64-1_direct",
    ],
)

cc_library(
    name = "add-64-1_bisection",
    srcs = ["add-64-1/add-64-1_bisection.cpp"],
    hdrs = ["add-64-1/add-64-1_bisection.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "add-64-1_openfhe",
    srcs = ["add-64-1/add-64-1_openfhe.cpp"],
    hdrs = ["add-64-1/add-64-1_openfhe.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "add-64-1_direct",
    srcs = ["add-64-1/add-64-1_direct.cpp"],
    hdrs = ["add-64-1/add-64-1_direct.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "main_add-64-2_common",
    srcs = ["add-64-2/main_add-64-2_common.cpp"],
    hdrs = ["add-64-2/main_add-64-2_common.h"],
    deps = [
        ":params",
        "@openfhe//:pke",
    ],
)

cc_binary(
    name = "main_add-64-2_bisection",
    srcs = ["add-64-2/main_add-64-2_bisection.cpp"],
    deps = [
        ":main_add-64-2_common",
        ":add-64-2_bisection",
    ],
)

cc_binary(
    name = "main_add-64-2_openfhe",
    srcs = ["add-64-2/main_add-64-2_openfhe.cpp"],
    deps = [
        ":main_add-64-2_common",
        ":add-64-2_openfhe",
    ],
)

cc_binary(
    name = "main_add-64-2_direct",
    srcs = ["add-64-2/main_add-64-2_direct.cpp"],
    deps = [
        ":main_add-64-2_common",
        ":add-64-2_direct",
    ],
)

cc_library(
    name = "add-64-2_bisection",
    srcs = ["add-64-2/add-64-2_bisection.cpp"],
    hdrs = ["add-64-2/add-64-2_bisection.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "add-64-2_openfhe",
    srcs = ["add-64-2/add-64-2_openfhe.cpp"],
    hdrs = ["add-64-2/add-64-2_openfhe.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "add-64-2_direct",
    srcs = ["add-64-2/add-64-2_direct.cpp"],
    hdrs = ["add-64-2/add-64-2_direct.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "main_add-64-3_common",
    srcs = ["add-64-3/main_add-64-3_common.cpp"],
    hdrs = ["add-64-3/main_add-64-3_common.h"],
    deps = [
        ":params",
        "@openfhe//:pke",
    ],
)

cc_binary(
    name = "main_add-64-3_bisection",
    srcs = ["add-64-3/main_add-64-3_bisection.cpp"],
    deps = [
        ":main_add-64-3_common",
        ":add-64-3_bisection",
    ],
)

cc_binary(
    name = "main_add-64-3_openfhe",
    srcs = ["add-64-3/main_add-64-3_openfhe.cpp"],
    deps = [
        ":main_add-64-3_common",
        ":add-64-3_openfhe",
    ],
)

cc_binary(
    name = "main_add-64-3_direct",
    srcs = ["add-64-3/main_add-64-3_direct.cpp"],
    deps = [
        ":main_add-64-3_common",
        ":add-64-3_direct",
    ],
)

cc_library(
    name = "add-64-3_bisection",
    srcs = ["add-64-3/add-64-3_bisection.cpp"],
    hdrs = ["add-64-3/add-64-3_bisection.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "add-64-3_openfhe",
    srcs = ["add-64-3/add-64-3_openfhe.cpp"],
    hdrs = ["add-64-3/add-64-3_openfhe.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "add-64-3_direct",
    srcs = ["add-64-3/add-64-3_direct.cpp"],
    hdrs = ["add-64-3/add-64-3_direct.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "main_add-64-4_common",
    srcs = ["add-64-4/main_add-64-4_common.cpp"],
    hdrs = ["add-64-4/main_add-64-4_common.h"],
    deps = [
        ":params",
        "@openfhe//:pke",
    ],
)

cc_binary(
    name = "main_add-64-4_bisection",
    srcs = ["add-64-4/main_add-64-4_bisection.cpp"],
    deps = [
        ":main_add-64-4_common",
        ":add-64-4_bisection",
    ],
)

cc_binary(
    name = "main_add-64-4_openfhe",
    srcs = ["add-64-4/main_add-64-4_openfhe.cpp"],
    deps = [
        ":main_add-64-4_common",
        ":add-64-4_openfhe",
    ],
)

cc_binary(
    name = "main_add-64-4_direct",
    srcs = ["add-64-4/main_add-64-4_direct.cpp"],
    deps = [
        ":main_add-64-4_common",
        ":add-64-4_direct",
    ],
)

cc_library(
    name = "add-64-4_bisection",
    srcs = ["add-64-4/add-64-4_bisection.cpp"],
    hdrs = ["add-64-4/add-64-4_bisection.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "add-64-4_openfhe",
    srcs = ["add-64-4/add-64-4_openfhe.cpp"],
    hdrs = ["add-64-4/add-64-4_openfhe.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "add-64-4_direct",
    srcs = ["add-64-4/add-64-4_direct.cpp"],
    hdrs = ["add-64-4/add-64-4_direct.h"],
    deps = [
        "@openfhe//:pke",
    ],
)
