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
    name = "main_dot8_common",
    srcs = ["dot8/main_dot8_common.cpp"],
    hdrs = ["dot8/main_dot8_common.h"],
    deps = [
        ":params",
        "@openfhe//:pke",
    ],
)

cc_binary(
    name = "main_dot8_bisection",
    srcs = ["dot8/main_dot8_bisection.cpp"],
    deps = [
        ":main_dot8_common",
        ":dot8_bisection",
    ],
)

cc_binary(
    name = "main_dot8_closed",
    srcs = ["dot8/main_dot8_closed.cpp"],
    deps = [
        ":main_dot8_common",
        ":dot8_closed",
    ],
)

cc_binary(
    name = "main_dot8_direct",
    srcs = ["dot8/main_dot8_direct.cpp"],
    deps = [
        ":main_dot8_common",
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
    name = "dot8_closed",
    srcs = ["dot8/dot8_closed.cpp"],
    hdrs = ["dot8/dot8_closed.h"],
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
    name = "main_psi_closed",
    srcs = ["psi/main_psi_closed.cpp"],
    deps = [
        ":psi_common",
        ":psi_closed",
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
    name = "psi_closed",
    srcs = ["psi/psi_closed.cpp"],
    hdrs = ["psi/psi_closed.h"],
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
    name = "main_mult5_closed",
    srcs = ["mult5/main_mult5_closed.cpp"],
    deps = [
        ":mult5_common",
        ":mult5_closed",
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
    name = "mult5_closed",
    srcs = ["mult5/mult5_closed.cpp"],
    hdrs = ["mult5/mult5_closed.h"],
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
    name = "main_add-64-0_closed",
    srcs = ["add-64-0/main_add-64-0_closed.cpp"],
    deps = [
        ":main_add-64-0_common",
        ":add-64-0_closed",
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
    name = "add-64-0_closed",
    srcs = ["add-64-0/add-64-0_closed.cpp"],
    hdrs = ["add-64-0/add-64-0_closed.h"],
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
    name = "main_add-64-1_closed",
    srcs = ["add-64-1/main_add-64-1_closed.cpp"],
    deps = [
        ":main_add-64-1_common",
        ":add-64-1_closed",
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
    name = "add-64-1_closed",
    srcs = ["add-64-1/add-64-1_closed.cpp"],
    hdrs = ["add-64-1/add-64-1_closed.h"],
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
    name = "main_add-64-2_closed",
    srcs = ["add-64-2/main_add-64-2_closed.cpp"],
    deps = [
        ":main_add-64-2_common",
        ":add-64-2_closed",
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
    name = "add-64-2_closed",
    srcs = ["add-64-2/add-64-2_closed.cpp"],
    hdrs = ["add-64-2/add-64-2_closed.h"],
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
    name = "main_add-64-3_closed",
    srcs = ["add-64-3/main_add-64-3_closed.cpp"],
    deps = [
        ":main_add-64-3_common",
        ":add-64-3_closed",
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
    name = "add-64-3_closed",
    srcs = ["add-64-3/add-64-3_closed.cpp"],
    hdrs = ["add-64-3/add-64-3_closed.h"],
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
    name = "main_add-64-4_closed",
    srcs = ["add-64-4/main_add-64-4_closed.cpp"],
    deps = [
        ":main_add-64-4_common",
        ":add-64-4_closed",
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
    name = "add-64-4_closed",
    srcs = ["add-64-4/add-64-4_closed.cpp"],
    hdrs = ["add-64-4/add-64-4_closed.h"],
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

cc_library(
    name = "main_add-depth-0_common",
    srcs = ["add-depth-0/main_add-depth-0_common.cpp"],
    hdrs = ["add-depth-0/main_add-depth-0_common.h"],
    deps = [
        ":params",
        "@openfhe//:pke",
    ],
)

cc_binary(
    name = "main_add-depth-0_bisection",
    srcs = ["add-depth-0/main_add-depth-0_bisection.cpp"],
    deps = [
        ":main_add-depth-0_common",
        ":add-depth-0_bisection",
    ],
)

cc_binary(
    name = "main_add-depth-0_closed",
    srcs = ["add-depth-0/main_add-depth-0_closed.cpp"],
    deps = [
        ":main_add-depth-0_common",
        ":add-depth-0_closed",
    ],
)

cc_binary(
    name = "main_add-depth-0_direct",
    srcs = ["add-depth-0/main_add-depth-0_direct.cpp"],
    deps = [
        ":main_add-depth-0_common",
        ":add-depth-0_direct",
    ],
)

cc_library(
    name = "add-depth-0_bisection",
    srcs = ["add-depth-0/add-depth-0_bisection.cpp"],
    hdrs = ["add-depth-0/add-depth-0_bisection.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "add-depth-0_closed",
    srcs = ["add-depth-0/add-depth-0_closed.cpp"],
    hdrs = ["add-depth-0/add-depth-0_closed.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "add-depth-0_direct",
    srcs = ["add-depth-0/add-depth-0_direct.cpp"],
    hdrs = ["add-depth-0/add-depth-0_direct.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "main_add-depth-1_common",
    srcs = ["add-depth-1/main_add-depth-1_common.cpp"],
    hdrs = ["add-depth-1/main_add-depth-1_common.h"],
    deps = [
        ":params",
        "@openfhe//:pke",
    ],
)

cc_binary(
    name = "main_add-depth-1_bisection",
    srcs = ["add-depth-1/main_add-depth-1_bisection.cpp"],
    deps = [
        ":main_add-depth-1_common",
        ":add-depth-1_bisection",
    ],
)

cc_binary(
    name = "main_add-depth-1_closed",
    srcs = ["add-depth-1/main_add-depth-1_closed.cpp"],
    deps = [
        ":main_add-depth-1_common",
        ":add-depth-1_closed",
    ],
)

cc_binary(
    name = "main_add-depth-1_direct",
    srcs = ["add-depth-1/main_add-depth-1_direct.cpp"],
    deps = [
        ":main_add-depth-1_common",
        ":add-depth-1_direct",
    ],
)

cc_library(
    name = "add-depth-1_bisection",
    srcs = ["add-depth-1/add-depth-1_bisection.cpp"],
    hdrs = ["add-depth-1/add-depth-1_bisection.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "add-depth-1_closed",
    srcs = ["add-depth-1/add-depth-1_closed.cpp"],
    hdrs = ["add-depth-1/add-depth-1_closed.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "add-depth-1_direct",
    srcs = ["add-depth-1/add-depth-1_direct.cpp"],
    hdrs = ["add-depth-1/add-depth-1_direct.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "main_add-depth-2_common",
    srcs = ["add-depth-2/main_add-depth-2_common.cpp"],
    hdrs = ["add-depth-2/main_add-depth-2_common.h"],
    deps = [
        ":params",
        "@openfhe//:pke",
    ],
)

cc_binary(
    name = "main_add-depth-2_bisection",
    srcs = ["add-depth-2/main_add-depth-2_bisection.cpp"],
    deps = [
        ":main_add-depth-2_common",
        ":add-depth-2_bisection",
    ],
)

cc_binary(
    name = "main_add-depth-2_closed",
    srcs = ["add-depth-2/main_add-depth-2_closed.cpp"],
    deps = [
        ":main_add-depth-2_common",
        ":add-depth-2_closed",
    ],
)

cc_binary(
    name = "main_add-depth-2_direct",
    srcs = ["add-depth-2/main_add-depth-2_direct.cpp"],
    deps = [
        ":main_add-depth-2_common",
        ":add-depth-2_direct",
    ],
)

cc_library(
    name = "add-depth-2_bisection",
    srcs = ["add-depth-2/add-depth-2_bisection.cpp"],
    hdrs = ["add-depth-2/add-depth-2_bisection.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "add-depth-2_closed",
    srcs = ["add-depth-2/add-depth-2_closed.cpp"],
    hdrs = ["add-depth-2/add-depth-2_closed.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "add-depth-2_direct",
    srcs = ["add-depth-2/add-depth-2_direct.cpp"],
    hdrs = ["add-depth-2/add-depth-2_direct.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "main_add-depth-3_common",
    srcs = ["add-depth-3/main_add-depth-3_common.cpp"],
    hdrs = ["add-depth-3/main_add-depth-3_common.h"],
    deps = [
        ":params",
        "@openfhe//:pke",
    ],
)

cc_binary(
    name = "main_add-depth-3_bisection",
    srcs = ["add-depth-3/main_add-depth-3_bisection.cpp"],
    deps = [
        ":main_add-depth-3_common",
        ":add-depth-3_bisection",
    ],
)

cc_binary(
    name = "main_add-depth-3_closed",
    srcs = ["add-depth-3/main_add-depth-3_closed.cpp"],
    deps = [
        ":main_add-depth-3_common",
        ":add-depth-3_closed",
    ],
)

cc_binary(
    name = "main_add-depth-3_direct",
    srcs = ["add-depth-3/main_add-depth-3_direct.cpp"],
    deps = [
        ":main_add-depth-3_common",
        ":add-depth-3_direct",
    ],
)

cc_library(
    name = "add-depth-3_bisection",
    srcs = ["add-depth-3/add-depth-3_bisection.cpp"],
    hdrs = ["add-depth-3/add-depth-3_bisection.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "add-depth-3_closed",
    srcs = ["add-depth-3/add-depth-3_closed.cpp"],
    hdrs = ["add-depth-3/add-depth-3_closed.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "add-depth-3_direct",
    srcs = ["add-depth-3/add-depth-3_direct.cpp"],
    hdrs = ["add-depth-3/add-depth-3_direct.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "main_add-depth-4_common",
    srcs = ["add-depth-4/main_add-depth-4_common.cpp"],
    hdrs = ["add-depth-4/main_add-depth-4_common.h"],
    deps = [
        ":params",
        "@openfhe//:pke",
    ],
)

cc_binary(
    name = "main_add-depth-4_bisection",
    srcs = ["add-depth-4/main_add-depth-4_bisection.cpp"],
    deps = [
        ":main_add-depth-4_common",
        ":add-depth-4_bisection",
    ],
)

cc_binary(
    name = "main_add-depth-4_closed",
    srcs = ["add-depth-4/main_add-depth-4_closed.cpp"],
    deps = [
        ":main_add-depth-4_common",
        ":add-depth-4_closed",
    ],
)

cc_binary(
    name = "main_add-depth-4_direct",
    srcs = ["add-depth-4/main_add-depth-4_direct.cpp"],
    deps = [
        ":main_add-depth-4_common",
        ":add-depth-4_direct",
    ],
)

cc_library(
    name = "add-depth-4_bisection",
    srcs = ["add-depth-4/add-depth-4_bisection.cpp"],
    hdrs = ["add-depth-4/add-depth-4_bisection.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "add-depth-4_closed",
    srcs = ["add-depth-4/add-depth-4_closed.cpp"],
    hdrs = ["add-depth-4/add-depth-4_closed.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "add-depth-4_direct",
    srcs = ["add-depth-4/add-depth-4_direct.cpp"],
    hdrs = ["add-depth-4/add-depth-4_direct.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "main_add-depth-5_common",
    srcs = ["add-depth-5/main_add-depth-5_common.cpp"],
    hdrs = ["add-depth-5/main_add-depth-5_common.h"],
    deps = [
        ":params",
        "@openfhe//:pke",
    ],
)

cc_binary(
    name = "main_add-depth-5_bisection",
    srcs = ["add-depth-5/main_add-depth-5_bisection.cpp"],
    deps = [
        ":main_add-depth-5_common",
        ":add-depth-5_bisection",
    ],
)

cc_binary(
    name = "main_add-depth-5_closed",
    srcs = ["add-depth-5/main_add-depth-5_closed.cpp"],
    deps = [
        ":main_add-depth-5_common",
        ":add-depth-5_closed",
    ],
)

cc_binary(
    name = "main_add-depth-5_direct",
    srcs = ["add-depth-5/main_add-depth-5_direct.cpp"],
    deps = [
        ":main_add-depth-5_common",
        ":add-depth-5_direct",
    ],
)

cc_library(
    name = "add-depth-5_bisection",
    srcs = ["add-depth-5/add-depth-5_bisection.cpp"],
    hdrs = ["add-depth-5/add-depth-5_bisection.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "add-depth-5_closed",
    srcs = ["add-depth-5/add-depth-5_closed.cpp"],
    hdrs = ["add-depth-5/add-depth-5_closed.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "add-depth-5_direct",
    srcs = ["add-depth-5/add-depth-5_direct.cpp"],
    hdrs = ["add-depth-5/add-depth-5_direct.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "main_add-depth-6_common",
    srcs = ["add-depth-6/main_add-depth-6_common.cpp"],
    hdrs = ["add-depth-6/main_add-depth-6_common.h"],
    deps = [
        ":params",
        "@openfhe//:pke",
    ],
)

cc_binary(
    name = "main_add-depth-6_bisection",
    srcs = ["add-depth-6/main_add-depth-6_bisection.cpp"],
    deps = [
        ":main_add-depth-6_common",
        ":add-depth-6_bisection",
    ],
)

cc_binary(
    name = "main_add-depth-6_closed",
    srcs = ["add-depth-6/main_add-depth-6_closed.cpp"],
    deps = [
        ":main_add-depth-6_common",
        ":add-depth-6_closed",
    ],
)

cc_binary(
    name = "main_add-depth-6_direct",
    srcs = ["add-depth-6/main_add-depth-6_direct.cpp"],
    deps = [
        ":main_add-depth-6_common",
        ":add-depth-6_direct",
    ],
)

cc_library(
    name = "add-depth-6_bisection",
    srcs = ["add-depth-6/add-depth-6_bisection.cpp"],
    hdrs = ["add-depth-6/add-depth-6_bisection.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "add-depth-6_closed",
    srcs = ["add-depth-6/add-depth-6_closed.cpp"],
    hdrs = ["add-depth-6/add-depth-6_closed.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "add-depth-6_direct",
    srcs = ["add-depth-6/add-depth-6_direct.cpp"],
    hdrs = ["add-depth-6/add-depth-6_direct.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "main_add-depth-7_common",
    srcs = ["add-depth-7/main_add-depth-7_common.cpp"],
    hdrs = ["add-depth-7/main_add-depth-7_common.h"],
    deps = [
        ":params",
        "@openfhe//:pke",
    ],
)

cc_binary(
    name = "main_add-depth-7_bisection",
    srcs = ["add-depth-7/main_add-depth-7_bisection.cpp"],
    deps = [
        ":main_add-depth-7_common",
        ":add-depth-7_bisection",
    ],
)

cc_binary(
    name = "main_add-depth-7_closed",
    srcs = ["add-depth-7/main_add-depth-7_closed.cpp"],
    deps = [
        ":main_add-depth-7_common",
        ":add-depth-7_closed",
    ],
)

cc_binary(
    name = "main_add-depth-7_direct",
    srcs = ["add-depth-7/main_add-depth-7_direct.cpp"],
    deps = [
        ":main_add-depth-7_common",
        ":add-depth-7_direct",
    ],
)

cc_library(
    name = "add-depth-7_bisection",
    srcs = ["add-depth-7/add-depth-7_bisection.cpp"],
    hdrs = ["add-depth-7/add-depth-7_bisection.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "add-depth-7_closed",
    srcs = ["add-depth-7/add-depth-7_closed.cpp"],
    hdrs = ["add-depth-7/add-depth-7_closed.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "add-depth-7_direct",
    srcs = ["add-depth-7/add-depth-7_direct.cpp"],
    hdrs = ["add-depth-7/add-depth-7_direct.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "main_add-depth-8_common",
    srcs = ["add-depth-8/main_add-depth-8_common.cpp"],
    hdrs = ["add-depth-8/main_add-depth-8_common.h"],
    deps = [
        ":params",
        "@openfhe//:pke",
    ],
)

cc_binary(
    name = "main_add-depth-8_bisection",
    srcs = ["add-depth-8/main_add-depth-8_bisection.cpp"],
    deps = [
        ":main_add-depth-8_common",
        ":add-depth-8_bisection",
    ],
)

cc_binary(
    name = "main_add-depth-8_closed",
    srcs = ["add-depth-8/main_add-depth-8_closed.cpp"],
    deps = [
        ":main_add-depth-8_common",
        ":add-depth-8_closed",
    ],
)

cc_binary(
    name = "main_add-depth-8_direct",
    srcs = ["add-depth-8/main_add-depth-8_direct.cpp"],
    deps = [
        ":main_add-depth-8_common",
        ":add-depth-8_direct",
    ],
)

cc_library(
    name = "add-depth-8_bisection",
    srcs = ["add-depth-8/add-depth-8_bisection.cpp"],
    hdrs = ["add-depth-8/add-depth-8_bisection.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "add-depth-8_closed",
    srcs = ["add-depth-8/add-depth-8_closed.cpp"],
    hdrs = ["add-depth-8/add-depth-8_closed.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "add-depth-8_direct",
    srcs = ["add-depth-8/add-depth-8_direct.cpp"],
    hdrs = ["add-depth-8/add-depth-8_direct.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "main_add-depth-9_common",
    srcs = ["add-depth-9/main_add-depth-9_common.cpp"],
    hdrs = ["add-depth-9/main_add-depth-9_common.h"],
    deps = [
        ":params",
        "@openfhe//:pke",
    ],
)

cc_binary(
    name = "main_add-depth-9_bisection",
    srcs = ["add-depth-9/main_add-depth-9_bisection.cpp"],
    deps = [
        ":main_add-depth-9_common",
        ":add-depth-9_bisection",
    ],
)

cc_binary(
    name = "main_add-depth-9_closed",
    srcs = ["add-depth-9/main_add-depth-9_closed.cpp"],
    deps = [
        ":main_add-depth-9_common",
        ":add-depth-9_closed",
    ],
)

cc_binary(
    name = "main_add-depth-9_direct",
    srcs = ["add-depth-9/main_add-depth-9_direct.cpp"],
    deps = [
        ":main_add-depth-9_common",
        ":add-depth-9_direct",
    ],
)

cc_library(
    name = "add-depth-9_bisection",
    srcs = ["add-depth-9/add-depth-9_bisection.cpp"],
    hdrs = ["add-depth-9/add-depth-9_bisection.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "add-depth-9_closed",
    srcs = ["add-depth-9/add-depth-9_closed.cpp"],
    hdrs = ["add-depth-9/add-depth-9_closed.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "add-depth-9_direct",
    srcs = ["add-depth-9/add-depth-9_direct.cpp"],
    hdrs = ["add-depth-9/add-depth-9_direct.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "main_add-depth-10_common",
    srcs = ["add-depth-10/main_add-depth-10_common.cpp"],
    hdrs = ["add-depth-10/main_add-depth-10_common.h"],
    deps = [
        ":params",
        "@openfhe//:pke",
    ],
)

cc_binary(
    name = "main_add-depth-10_bisection",
    srcs = ["add-depth-10/main_add-depth-10_bisection.cpp"],
    deps = [
        ":main_add-depth-10_common",
        ":add-depth-10_bisection",
    ],
)

cc_binary(
    name = "main_add-depth-10_closed",
    srcs = ["add-depth-10/main_add-depth-10_closed.cpp"],
    deps = [
        ":main_add-depth-10_common",
        ":add-depth-10_closed",
    ],
)

cc_binary(
    name = "main_add-depth-10_direct",
    srcs = ["add-depth-10/main_add-depth-10_direct.cpp"],
    deps = [
        ":main_add-depth-10_common",
        ":add-depth-10_direct",
    ],
)

cc_library(
    name = "add-depth-10_bisection",
    srcs = ["add-depth-10/add-depth-10_bisection.cpp"],
    hdrs = ["add-depth-10/add-depth-10_bisection.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "add-depth-10_closed",
    srcs = ["add-depth-10/add-depth-10_closed.cpp"],
    hdrs = ["add-depth-10/add-depth-10_closed.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "add-depth-10_direct",
    srcs = ["add-depth-10/add-depth-10_direct.cpp"],
    hdrs = ["add-depth-10/add-depth-10_direct.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "main_add-depth-11_common",
    srcs = ["add-depth-11/main_add-depth-11_common.cpp"],
    hdrs = ["add-depth-11/main_add-depth-11_common.h"],
    deps = [
        ":params",
        "@openfhe//:pke",
    ],
)

cc_binary(
    name = "main_add-depth-11_bisection",
    srcs = ["add-depth-11/main_add-depth-11_bisection.cpp"],
    deps = [
        ":main_add-depth-11_common",
        ":add-depth-11_bisection",
    ],
)

cc_binary(
    name = "main_add-depth-11_closed",
    srcs = ["add-depth-11/main_add-depth-11_closed.cpp"],
    deps = [
        ":main_add-depth-11_common",
        ":add-depth-11_closed",
    ],
)

cc_binary(
    name = "main_add-depth-11_direct",
    srcs = ["add-depth-11/main_add-depth-11_direct.cpp"],
    deps = [
        ":main_add-depth-11_common",
        ":add-depth-11_direct",
    ],
)

cc_library(
    name = "add-depth-11_bisection",
    srcs = ["add-depth-11/add-depth-11_bisection.cpp"],
    hdrs = ["add-depth-11/add-depth-11_bisection.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "add-depth-11_closed",
    srcs = ["add-depth-11/add-depth-11_closed.cpp"],
    hdrs = ["add-depth-11/add-depth-11_closed.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "add-depth-11_direct",
    srcs = ["add-depth-11/add-depth-11_direct.cpp"],
    hdrs = ["add-depth-11/add-depth-11_direct.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "main_add-depth-12_common",
    srcs = ["add-depth-12/main_add-depth-12_common.cpp"],
    hdrs = ["add-depth-12/main_add-depth-12_common.h"],
    deps = [
        ":params",
        "@openfhe//:pke",
    ],
)

cc_binary(
    name = "main_add-depth-12_bisection",
    srcs = ["add-depth-12/main_add-depth-12_bisection.cpp"],
    deps = [
        ":main_add-depth-12_common",
        ":add-depth-12_bisection",
    ],
)

cc_binary(
    name = "main_add-depth-12_closed",
    srcs = ["add-depth-12/main_add-depth-12_closed.cpp"],
    deps = [
        ":main_add-depth-12_common",
        ":add-depth-12_closed",
    ],
)

cc_binary(
    name = "main_add-depth-12_direct",
    srcs = ["add-depth-12/main_add-depth-12_direct.cpp"],
    deps = [
        ":main_add-depth-12_common",
        ":add-depth-12_direct",
    ],
)

cc_library(
    name = "add-depth-12_bisection",
    srcs = ["add-depth-12/add-depth-12_bisection.cpp"],
    hdrs = ["add-depth-12/add-depth-12_bisection.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "add-depth-12_closed",
    srcs = ["add-depth-12/add-depth-12_closed.cpp"],
    hdrs = ["add-depth-12/add-depth-12_closed.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "add-depth-12_direct",
    srcs = ["add-depth-12/add-depth-12_direct.cpp"],
    hdrs = ["add-depth-12/add-depth-12_direct.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "main_rotate-1_common",
    srcs = ["rotate-1/main_rotate-1_common.cpp"],
    hdrs = ["rotate-1/main_rotate-1_common.h"],
    deps = [
        ":params",
        "@openfhe//:pke",
    ],
)

cc_binary(
    name = "main_rotate-1_bisection",
    srcs = ["rotate-1/main_rotate-1_bisection.cpp"],
    deps = [
        ":main_rotate-1_common",
        ":rotate-1_bisection",
    ],
)

cc_binary(
    name = "main_rotate-1_closed",
    srcs = ["rotate-1/main_rotate-1_closed.cpp"],
    deps = [
        ":main_rotate-1_common",
        ":rotate-1_closed",
    ],
)

cc_binary(
    name = "main_rotate-1_direct",
    srcs = ["rotate-1/main_rotate-1_direct.cpp"],
    deps = [
        ":main_rotate-1_common",
        ":rotate-1_direct",
    ],
)

cc_library(
    name = "rotate-1_bisection",
    srcs = ["rotate-1/rotate-1_bisection.cpp"],
    hdrs = ["rotate-1/rotate-1_bisection.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "rotate-1_closed",
    srcs = ["rotate-1/rotate-1_closed.cpp"],
    hdrs = ["rotate-1/rotate-1_closed.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "rotate-1_direct",
    srcs = ["rotate-1/rotate-1_direct.cpp"],
    hdrs = ["rotate-1/rotate-1_direct.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "main_rotate-2_common",
    srcs = ["rotate-2/main_rotate-2_common.cpp"],
    hdrs = ["rotate-2/main_rotate-2_common.h"],
    deps = [
        ":params",
        "@openfhe//:pke",
    ],
)

cc_binary(
    name = "main_rotate-2_bisection",
    srcs = ["rotate-2/main_rotate-2_bisection.cpp"],
    deps = [
        ":main_rotate-2_common",
        ":rotate-2_bisection",
    ],
)

cc_binary(
    name = "main_rotate-2_closed",
    srcs = ["rotate-2/main_rotate-2_closed.cpp"],
    deps = [
        ":main_rotate-2_common",
        ":rotate-2_closed",
    ],
)

cc_binary(
    name = "main_rotate-2_direct",
    srcs = ["rotate-2/main_rotate-2_direct.cpp"],
    deps = [
        ":main_rotate-2_common",
        ":rotate-2_direct",
    ],
)

cc_library(
    name = "rotate-2_bisection",
    srcs = ["rotate-2/rotate-2_bisection.cpp"],
    hdrs = ["rotate-2/rotate-2_bisection.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "rotate-2_closed",
    srcs = ["rotate-2/rotate-2_closed.cpp"],
    hdrs = ["rotate-2/rotate-2_closed.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "rotate-2_direct",
    srcs = ["rotate-2/rotate-2_direct.cpp"],
    hdrs = ["rotate-2/rotate-2_direct.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "main_rotate-3_common",
    srcs = ["rotate-3/main_rotate-3_common.cpp"],
    hdrs = ["rotate-3/main_rotate-3_common.h"],
    deps = [
        ":params",
        "@openfhe//:pke",
    ],
)

cc_binary(
    name = "main_rotate-3_bisection",
    srcs = ["rotate-3/main_rotate-3_bisection.cpp"],
    deps = [
        ":main_rotate-3_common",
        ":rotate-3_bisection",
    ],
)

cc_binary(
    name = "main_rotate-3_closed",
    srcs = ["rotate-3/main_rotate-3_closed.cpp"],
    deps = [
        ":main_rotate-3_common",
        ":rotate-3_closed",
    ],
)

cc_binary(
    name = "main_rotate-3_direct",
    srcs = ["rotate-3/main_rotate-3_direct.cpp"],
    deps = [
        ":main_rotate-3_common",
        ":rotate-3_direct",
    ],
)

cc_library(
    name = "rotate-3_bisection",
    srcs = ["rotate-3/rotate-3_bisection.cpp"],
    hdrs = ["rotate-3/rotate-3_bisection.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "rotate-3_closed",
    srcs = ["rotate-3/rotate-3_closed.cpp"],
    hdrs = ["rotate-3/rotate-3_closed.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "rotate-3_direct",
    srcs = ["rotate-3/rotate-3_direct.cpp"],
    hdrs = ["rotate-3/rotate-3_direct.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "main_rotate-4_common",
    srcs = ["rotate-4/main_rotate-4_common.cpp"],
    hdrs = ["rotate-4/main_rotate-4_common.h"],
    deps = [
        ":params",
        "@openfhe//:pke",
    ],
)

cc_binary(
    name = "main_rotate-4_bisection",
    srcs = ["rotate-4/main_rotate-4_bisection.cpp"],
    deps = [
        ":main_rotate-4_common",
        ":rotate-4_bisection",
    ],
)

cc_binary(
    name = "main_rotate-4_closed",
    srcs = ["rotate-4/main_rotate-4_closed.cpp"],
    deps = [
        ":main_rotate-4_common",
        ":rotate-4_closed",
    ],
)

cc_binary(
    name = "main_rotate-4_direct",
    srcs = ["rotate-4/main_rotate-4_direct.cpp"],
    deps = [
        ":main_rotate-4_common",
        ":rotate-4_direct",
    ],
)

cc_library(
    name = "rotate-4_bisection",
    srcs = ["rotate-4/rotate-4_bisection.cpp"],
    hdrs = ["rotate-4/rotate-4_bisection.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "rotate-4_closed",
    srcs = ["rotate-4/rotate-4_closed.cpp"],
    hdrs = ["rotate-4/rotate-4_closed.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "rotate-4_direct",
    srcs = ["rotate-4/rotate-4_direct.cpp"],
    hdrs = ["rotate-4/rotate-4_direct.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "main_rotate-0_common",
    srcs = ["rotate-0/main_rotate-0_common.cpp"],
    hdrs = ["rotate-0/main_rotate-0_common.h"],
    deps = [
        ":params",
        "@openfhe//:pke",
    ],
)

cc_binary(
    name = "main_rotate-0_bisection",
    srcs = ["rotate-0/main_rotate-0_bisection.cpp"],
    deps = [
        ":main_rotate-0_common",
        ":rotate-0_bisection",
    ],
)

cc_binary(
    name = "main_rotate-0_closed",
    srcs = ["rotate-0/main_rotate-0_closed.cpp"],
    deps = [
        ":main_rotate-0_common",
        ":rotate-0_closed",
    ],
)

cc_binary(
    name = "main_rotate-0_direct",
    srcs = ["rotate-0/main_rotate-0_direct.cpp"],
    deps = [
        ":main_rotate-0_common",
        ":rotate-0_direct",
    ],
)

cc_library(
    name = "rotate-0_bisection",
    srcs = ["rotate-0/rotate-0_bisection.cpp"],
    hdrs = ["rotate-0/rotate-0_bisection.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "rotate-0_closed",
    srcs = ["rotate-0/rotate-0_closed.cpp"],
    hdrs = ["rotate-0/rotate-0_closed.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "rotate-0_direct",
    srcs = ["rotate-0/rotate-0_direct.cpp"],
    hdrs = ["rotate-0/rotate-0_direct.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "main_form-hill_common",
    srcs = ["form-hill/main_form-hill_common.cpp"],
    hdrs = ["form-hill/main_form-hill_common.h"],
    deps = [
        ":params",
        "@openfhe//:pke",
    ],
)

cc_binary(
    name = "main_form-hill_bisection",
    srcs = ["form-hill/main_form-hill_bisection.cpp"],
    deps = [
        ":main_form-hill_common",
        ":form-hill_bisection",
    ],
)

cc_binary(
    name = "main_form-hill_closed",
    srcs = ["form-hill/main_form-hill_closed.cpp"],
    deps = [
        ":main_form-hill_common",
        ":form-hill_closed",
    ],
)

cc_binary(
    name = "main_form-hill_direct",
    srcs = ["form-hill/main_form-hill_direct.cpp"],
    deps = [
        ":main_form-hill_common",
        ":form-hill_direct",
    ],
)

cc_library(
    name = "form-hill_bisection",
    srcs = ["form-hill/form-hill_bisection.cpp"],
    hdrs = ["form-hill/form-hill_bisection.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "form-hill_closed",
    srcs = ["form-hill/form-hill_closed.cpp"],
    hdrs = ["form-hill/form-hill_closed.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "form-hill_direct",
    srcs = ["form-hill/form-hill_direct.cpp"],
    hdrs = ["form-hill/form-hill_direct.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "main_form-high-low_common",
    srcs = ["form-high-low/main_form-high-low_common.cpp"],
    hdrs = ["form-high-low/main_form-high-low_common.h"],
    deps = [
        ":params",
        "@openfhe//:pke",
    ],
)

cc_binary(
    name = "main_form-high-low_bisection",
    srcs = ["form-high-low/main_form-high-low_bisection.cpp"],
    deps = [
        ":main_form-high-low_common",
        ":form-high-low_bisection",
    ],
)

cc_binary(
    name = "main_form-high-low_closed",
    srcs = ["form-high-low/main_form-high-low_closed.cpp"],
    deps = [
        ":main_form-high-low_common",
        ":form-high-low_closed",
    ],
)

cc_binary(
    name = "main_form-high-low_direct",
    srcs = ["form-high-low/main_form-high-low_direct.cpp"],
    deps = [
        ":main_form-high-low_common",
        ":form-high-low_direct",
    ],
)

cc_library(
    name = "form-high-low_bisection",
    srcs = ["form-high-low/form-high-low_bisection.cpp"],
    hdrs = ["form-high-low/form-high-low_bisection.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "form-high-low_closed",
    srcs = ["form-high-low/form-high-low_closed.cpp"],
    hdrs = ["form-high-low/form-high-low_closed.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "form-high-low_direct",
    srcs = ["form-high-low/form-high-low_direct.cpp"],
    hdrs = ["form-high-low/form-high-low_direct.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "main_form-low-high_common",
    srcs = ["form-low-high/main_form-low-high_common.cpp"],
    hdrs = ["form-low-high/main_form-low-high_common.h"],
    deps = [
        ":params",
        "@openfhe//:pke",
    ],
)

cc_binary(
    name = "main_form-low-high_bisection",
    srcs = ["form-low-high/main_form-low-high_bisection.cpp"],
    deps = [
        ":main_form-low-high_common",
        ":form-low-high_bisection",
    ],
)

cc_binary(
    name = "main_form-low-high_closed",
    srcs = ["form-low-high/main_form-low-high_closed.cpp"],
    deps = [
        ":main_form-low-high_common",
        ":form-low-high_closed",
    ],
)

cc_binary(
    name = "main_form-low-high_direct",
    srcs = ["form-low-high/main_form-low-high_direct.cpp"],
    deps = [
        ":main_form-low-high_common",
        ":form-low-high_direct",
    ],
)

cc_library(
    name = "form-low-high_bisection",
    srcs = ["form-low-high/form-low-high_bisection.cpp"],
    hdrs = ["form-low-high/form-low-high_bisection.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "form-low-high_closed",
    srcs = ["form-low-high/form-low-high_closed.cpp"],
    hdrs = ["form-low-high/form-low-high_closed.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "form-low-high_direct",
    srcs = ["form-low-high/form-low-high_direct.cpp"],
    hdrs = ["form-low-high/form-low-high_direct.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "main_form-decreasing_common",
    srcs = ["form-decreasing/main_form-decreasing_common.cpp"],
    hdrs = ["form-decreasing/main_form-decreasing_common.h"],
    deps = [
        ":params",
        "@openfhe//:pke",
    ],
)

cc_binary(
    name = "main_form-decreasing_bisection",
    srcs = ["form-decreasing/main_form-decreasing_bisection.cpp"],
    deps = [
        ":main_form-decreasing_common",
        ":form-decreasing_bisection",
    ],
)

cc_binary(
    name = "main_form-decreasing_closed",
    srcs = ["form-decreasing/main_form-decreasing_closed.cpp"],
    deps = [
        ":main_form-decreasing_common",
        ":form-decreasing_closed",
    ],
)

cc_binary(
    name = "main_form-decreasing_direct",
    srcs = ["form-decreasing/main_form-decreasing_direct.cpp"],
    deps = [
        ":main_form-decreasing_common",
        ":form-decreasing_direct",
    ],
)

cc_library(
    name = "form-decreasing_bisection",
    srcs = ["form-decreasing/form-decreasing_bisection.cpp"],
    hdrs = ["form-decreasing/form-decreasing_bisection.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "form-decreasing_closed",
    srcs = ["form-decreasing/form-decreasing_closed.cpp"],
    hdrs = ["form-decreasing/form-decreasing_closed.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "form-decreasing_direct",
    srcs = ["form-decreasing/form-decreasing_direct.cpp"],
    hdrs = ["form-decreasing/form-decreasing_direct.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "main_form-increasing_common",
    srcs = ["form-increasing/main_form-increasing_common.cpp"],
    hdrs = ["form-increasing/main_form-increasing_common.h"],
    deps = [
        ":params",
        "@openfhe//:pke",
    ],
)

cc_binary(
    name = "main_form-increasing_bisection",
    srcs = ["form-increasing/main_form-increasing_bisection.cpp"],
    deps = [
        ":main_form-increasing_common",
        ":form-increasing_bisection",
    ],
)

cc_binary(
    name = "main_form-increasing_closed",
    srcs = ["form-increasing/main_form-increasing_closed.cpp"],
    deps = [
        ":main_form-increasing_common",
        ":form-increasing_closed",
    ],
)

cc_binary(
    name = "main_form-increasing_direct",
    srcs = ["form-increasing/main_form-increasing_direct.cpp"],
    deps = [
        ":main_form-increasing_common",
        ":form-increasing_direct",
    ],
)

cc_library(
    name = "form-increasing_bisection",
    srcs = ["form-increasing/form-increasing_bisection.cpp"],
    hdrs = ["form-increasing/form-increasing_bisection.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "form-increasing_closed",
    srcs = ["form-increasing/form-increasing_closed.cpp"],
    hdrs = ["form-increasing/form-increasing_closed.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "form-increasing_direct",
    srcs = ["form-increasing/form-increasing_direct.cpp"],
    hdrs = ["form-increasing/form-increasing_direct.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "main_form-valley_common",
    srcs = ["form-valley/main_form-valley_common.cpp"],
    hdrs = ["form-valley/main_form-valley_common.h"],
    deps = [
        ":params",
        "@openfhe//:pke",
    ],
)

cc_binary(
    name = "main_form-valley_bisection",
    srcs = ["form-valley/main_form-valley_bisection.cpp"],
    deps = [
        ":main_form-valley_common",
        ":form-valley_bisection",
    ],
)

cc_binary(
    name = "main_form-valley_closed",
    srcs = ["form-valley/main_form-valley_closed.cpp"],
    deps = [
        ":main_form-valley_common",
        ":form-valley_closed",
    ],
)

cc_binary(
    name = "main_form-valley_direct",
    srcs = ["form-valley/main_form-valley_direct.cpp"],
    deps = [
        ":main_form-valley_common",
        ":form-valley_direct",
    ],
)

cc_library(
    name = "form-valley_bisection",
    srcs = ["form-valley/form-valley_bisection.cpp"],
    hdrs = ["form-valley/form-valley_bisection.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "form-valley_closed",
    srcs = ["form-valley/form-valley_closed.cpp"],
    hdrs = ["form-valley/form-valley_closed.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "form-valley_direct",
    srcs = ["form-valley/form-valley_direct.cpp"],
    hdrs = ["form-valley/form-valley_direct.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "main_form-equal_common",
    srcs = ["form-equal/main_form-equal_common.cpp"],
    hdrs = ["form-equal/main_form-equal_common.h"],
    deps = [
        ":params",
        "@openfhe//:pke",
    ],
)

cc_binary(
    name = "main_form-equal_bisection",
    srcs = ["form-equal/main_form-equal_bisection.cpp"],
    deps = [
        ":main_form-equal_common",
        ":form-equal_bisection",
    ],
)

cc_binary(
    name = "main_form-equal_closed",
    srcs = ["form-equal/main_form-equal_closed.cpp"],
    deps = [
        ":main_form-equal_common",
        ":form-equal_closed",
    ],
)

cc_binary(
    name = "main_form-equal_direct",
    srcs = ["form-equal/main_form-equal_direct.cpp"],
    deps = [
        ":main_form-equal_common",
        ":form-equal_direct",
    ],
)

cc_library(
    name = "form-equal_bisection",
    srcs = ["form-equal/form-equal_bisection.cpp"],
    hdrs = ["form-equal/form-equal_bisection.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "form-equal_closed",
    srcs = ["form-equal/form-equal_closed.cpp"],
    hdrs = ["form-equal/form-equal_closed.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "form-equal_direct",
    srcs = ["form-equal/form-equal_direct.cpp"],
    hdrs = ["form-equal/form-equal_direct.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "main_add-eq-3_common",
    srcs = ["add-eq-3/main_add-eq-3_common.cpp"],
    hdrs = ["add-eq-3/main_add-eq-3_common.h"],
    deps = [
        ":params",
        "@openfhe//:pke",
    ],
)

cc_binary(
    name = "main_add-eq-3_bisection",
    srcs = ["add-eq-3/main_add-eq-3_bisection.cpp"],
    deps = [
        ":main_add-eq-3_common",
        ":add-eq-3_bisection",
    ],
)

cc_binary(
    name = "main_add-eq-3_closed",
    srcs = ["add-eq-3/main_add-eq-3_closed.cpp"],
    deps = [
        ":main_add-eq-3_common",
        ":add-eq-3_closed",
    ],
)

cc_binary(
    name = "main_add-eq-3_direct",
    srcs = ["add-eq-3/main_add-eq-3_direct.cpp"],
    deps = [
        ":main_add-eq-3_common",
        ":add-eq-3_direct",
    ],
)

cc_library(
    name = "add-eq-3_bisection",
    srcs = ["add-eq-3/add-eq-3_bisection.cpp"],
    hdrs = ["add-eq-3/add-eq-3_bisection.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "add-eq-3_closed",
    srcs = ["add-eq-3/add-eq-3_closed.cpp"],
    hdrs = ["add-eq-3/add-eq-3_closed.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "add-eq-3_direct",
    srcs = ["add-eq-3/add-eq-3_direct.cpp"],
    hdrs = ["add-eq-3/add-eq-3_direct.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "main_add-eq-2_common",
    srcs = ["add-eq-2/main_add-eq-2_common.cpp"],
    hdrs = ["add-eq-2/main_add-eq-2_common.h"],
    deps = [
        ":params",
        "@openfhe//:pke",
    ],
)

cc_binary(
    name = "main_add-eq-2_bisection",
    srcs = ["add-eq-2/main_add-eq-2_bisection.cpp"],
    deps = [
        ":main_add-eq-2_common",
        ":add-eq-2_bisection",
    ],
)

cc_binary(
    name = "main_add-eq-2_closed",
    srcs = ["add-eq-2/main_add-eq-2_closed.cpp"],
    deps = [
        ":main_add-eq-2_common",
        ":add-eq-2_closed",
    ],
)

cc_binary(
    name = "main_add-eq-2_direct",
    srcs = ["add-eq-2/main_add-eq-2_direct.cpp"],
    deps = [
        ":main_add-eq-2_common",
        ":add-eq-2_direct",
    ],
)

cc_library(
    name = "add-eq-2_bisection",
    srcs = ["add-eq-2/add-eq-2_bisection.cpp"],
    hdrs = ["add-eq-2/add-eq-2_bisection.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "add-eq-2_closed",
    srcs = ["add-eq-2/add-eq-2_closed.cpp"],
    hdrs = ["add-eq-2/add-eq-2_closed.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "add-eq-2_direct",
    srcs = ["add-eq-2/add-eq-2_direct.cpp"],
    hdrs = ["add-eq-2/add-eq-2_direct.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "main_add-eq-1_common",
    srcs = ["add-eq-1/main_add-eq-1_common.cpp"],
    hdrs = ["add-eq-1/main_add-eq-1_common.h"],
    deps = [
        ":params",
        "@openfhe//:pke",
    ],
)

cc_binary(
    name = "main_add-eq-1_bisection",
    srcs = ["add-eq-1/main_add-eq-1_bisection.cpp"],
    deps = [
        ":main_add-eq-1_common",
        ":add-eq-1_bisection",
    ],
)

cc_binary(
    name = "main_add-eq-1_closed",
    srcs = ["add-eq-1/main_add-eq-1_closed.cpp"],
    deps = [
        ":main_add-eq-1_common",
        ":add-eq-1_closed",
    ],
)

cc_binary(
    name = "main_add-eq-1_direct",
    srcs = ["add-eq-1/main_add-eq-1_direct.cpp"],
    deps = [
        ":main_add-eq-1_common",
        ":add-eq-1_direct",
    ],
)

cc_library(
    name = "add-eq-1_bisection",
    srcs = ["add-eq-1/add-eq-1_bisection.cpp"],
    hdrs = ["add-eq-1/add-eq-1_bisection.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "add-eq-1_closed",
    srcs = ["add-eq-1/add-eq-1_closed.cpp"],
    hdrs = ["add-eq-1/add-eq-1_closed.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "add-eq-1_direct",
    srcs = ["add-eq-1/add-eq-1_direct.cpp"],
    hdrs = ["add-eq-1/add-eq-1_direct.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "main_add-eq-4_common",
    srcs = ["add-eq-4/main_add-eq-4_common.cpp"],
    hdrs = ["add-eq-4/main_add-eq-4_common.h"],
    deps = [
        ":params",
        "@openfhe//:pke",
    ],
)

cc_binary(
    name = "main_add-eq-4_bisection",
    srcs = ["add-eq-4/main_add-eq-4_bisection.cpp"],
    deps = [
        ":main_add-eq-4_common",
        ":add-eq-4_bisection",
    ],
)

cc_binary(
    name = "main_add-eq-4_closed",
    srcs = ["add-eq-4/main_add-eq-4_closed.cpp"],
    deps = [
        ":main_add-eq-4_common",
        ":add-eq-4_closed",
    ],
)

cc_binary(
    name = "main_add-eq-4_direct",
    srcs = ["add-eq-4/main_add-eq-4_direct.cpp"],
    deps = [
        ":main_add-eq-4_common",
        ":add-eq-4_direct",
    ],
)

cc_library(
    name = "add-eq-4_bisection",
    srcs = ["add-eq-4/add-eq-4_bisection.cpp"],
    hdrs = ["add-eq-4/add-eq-4_bisection.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "add-eq-4_closed",
    srcs = ["add-eq-4/add-eq-4_closed.cpp"],
    hdrs = ["add-eq-4/add-eq-4_closed.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "add-eq-4_direct",
    srcs = ["add-eq-4/add-eq-4_direct.cpp"],
    hdrs = ["add-eq-4/add-eq-4_direct.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "main_add-eq-5_common",
    srcs = ["add-eq-5/main_add-eq-5_common.cpp"],
    hdrs = ["add-eq-5/main_add-eq-5_common.h"],
    deps = [
        ":params",
        "@openfhe//:pke",
    ],
)

cc_binary(
    name = "main_add-eq-5_bisection",
    srcs = ["add-eq-5/main_add-eq-5_bisection.cpp"],
    deps = [
        ":main_add-eq-5_common",
        ":add-eq-5_bisection",
    ],
)

cc_binary(
    name = "main_add-eq-5_closed",
    srcs = ["add-eq-5/main_add-eq-5_closed.cpp"],
    deps = [
        ":main_add-eq-5_common",
        ":add-eq-5_closed",
    ],
)

cc_binary(
    name = "main_add-eq-5_direct",
    srcs = ["add-eq-5/main_add-eq-5_direct.cpp"],
    deps = [
        ":main_add-eq-5_common",
        ":add-eq-5_direct",
    ],
)

cc_library(
    name = "add-eq-5_bisection",
    srcs = ["add-eq-5/add-eq-5_bisection.cpp"],
    hdrs = ["add-eq-5/add-eq-5_bisection.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "add-eq-5_closed",
    srcs = ["add-eq-5/add-eq-5_closed.cpp"],
    hdrs = ["add-eq-5/add-eq-5_closed.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "add-eq-5_direct",
    srcs = ["add-eq-5/add-eq-5_direct.cpp"],
    hdrs = ["add-eq-5/add-eq-5_direct.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "main_add-eq-6_common",
    srcs = ["add-eq-6/main_add-eq-6_common.cpp"],
    hdrs = ["add-eq-6/main_add-eq-6_common.h"],
    deps = [
        ":params",
        "@openfhe//:pke",
    ],
)

cc_binary(
    name = "main_add-eq-6_bisection",
    srcs = ["add-eq-6/main_add-eq-6_bisection.cpp"],
    deps = [
        ":main_add-eq-6_common",
        ":add-eq-6_bisection",
    ],
)

cc_binary(
    name = "main_add-eq-6_closed",
    srcs = ["add-eq-6/main_add-eq-6_closed.cpp"],
    deps = [
        ":main_add-eq-6_common",
        ":add-eq-6_closed",
    ],
)

cc_binary(
    name = "main_add-eq-6_direct",
    srcs = ["add-eq-6/main_add-eq-6_direct.cpp"],
    deps = [
        ":main_add-eq-6_common",
        ":add-eq-6_direct",
    ],
)

cc_library(
    name = "add-eq-6_bisection",
    srcs = ["add-eq-6/add-eq-6_bisection.cpp"],
    hdrs = ["add-eq-6/add-eq-6_bisection.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "add-eq-6_closed",
    srcs = ["add-eq-6/add-eq-6_closed.cpp"],
    hdrs = ["add-eq-6/add-eq-6_closed.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "add-eq-6_direct",
    srcs = ["add-eq-6/add-eq-6_direct.cpp"],
    hdrs = ["add-eq-6/add-eq-6_direct.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "main_add-eq-0_common",
    srcs = ["add-eq-0/main_add-eq-0_common.cpp"],
    hdrs = ["add-eq-0/main_add-eq-0_common.h"],
    deps = [
        ":params",
        "@openfhe//:pke",
    ],
)

cc_binary(
    name = "main_add-eq-0_bisection",
    srcs = ["add-eq-0/main_add-eq-0_bisection.cpp"],
    deps = [
        ":main_add-eq-0_common",
        ":add-eq-0_bisection",
    ],
)

cc_binary(
    name = "main_add-eq-0_closed",
    srcs = ["add-eq-0/main_add-eq-0_closed.cpp"],
    deps = [
        ":main_add-eq-0_common",
        ":add-eq-0_closed",
    ],
)

cc_binary(
    name = "main_add-eq-0_direct",
    srcs = ["add-eq-0/main_add-eq-0_direct.cpp"],
    deps = [
        ":main_add-eq-0_common",
        ":add-eq-0_direct",
    ],
)

cc_library(
    name = "add-eq-0_bisection",
    srcs = ["add-eq-0/add-eq-0_bisection.cpp"],
    hdrs = ["add-eq-0/add-eq-0_bisection.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "add-eq-0_closed",
    srcs = ["add-eq-0/add-eq-0_closed.cpp"],
    hdrs = ["add-eq-0/add-eq-0_closed.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "add-eq-0_direct",
    srcs = ["add-eq-0/add-eq-0_direct.cpp"],
    hdrs = ["add-eq-0/add-eq-0_direct.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "main_add-end_common",
    srcs = ["add-end/main_add-end_common.cpp"],
    hdrs = ["add-end/main_add-end_common.h"],
    deps = [
        ":params",
        "@openfhe//:pke",
    ],
)

cc_binary(
    name = "main_add-end_bisection",
    srcs = ["add-end/main_add-end_bisection.cpp"],
    deps = [
        ":main_add-end_common",
        ":add-end_bisection",
    ],
)

cc_binary(
    name = "main_add-end_closed",
    srcs = ["add-end/main_add-end_closed.cpp"],
    deps = [
        ":main_add-end_common",
        ":add-end_closed",
    ],
)

cc_binary(
    name = "main_add-end_direct",
    srcs = ["add-end/main_add-end_direct.cpp"],
    deps = [
        ":main_add-end_common",
        ":add-end_direct",
    ],
)

cc_library(
    name = "add-end_bisection",
    srcs = ["add-end/add-end_bisection.cpp"],
    hdrs = ["add-end/add-end_bisection.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "add-end_closed",
    srcs = ["add-end/add-end_closed.cpp"],
    hdrs = ["add-end/add-end_closed.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "add-end_direct",
    srcs = ["add-end/add-end_direct.cpp"],
    hdrs = ["add-end/add-end_direct.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "main_add-256-4_common",
    srcs = ["add-256-4/main_add-256-4_common.cpp"],
    hdrs = ["add-256-4/main_add-256-4_common.h"],
    deps = [
        ":params",
        "@openfhe//:pke",
    ],
)

cc_binary(
    name = "main_add-256-4_bisection",
    srcs = ["add-256-4/main_add-256-4_bisection.cpp"],
    deps = [
        ":main_add-256-4_common",
        ":add-256-4_bisection",
    ],
)

cc_binary(
    name = "main_add-256-4_closed",
    srcs = ["add-256-4/main_add-256-4_closed.cpp"],
    deps = [
        ":main_add-256-4_common",
        ":add-256-4_closed",
    ],
)

cc_binary(
    name = "main_add-256-4_direct",
    srcs = ["add-256-4/main_add-256-4_direct.cpp"],
    deps = [
        ":main_add-256-4_common",
        ":add-256-4_direct",
    ],
)

cc_library(
    name = "add-256-4_bisection",
    srcs = ["add-256-4/add-256-4_bisection.cpp"],
    hdrs = ["add-256-4/add-256-4_bisection.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "add-256-4_closed",
    srcs = ["add-256-4/add-256-4_closed.cpp"],
    hdrs = ["add-256-4/add-256-4_closed.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "add-256-4_direct",
    srcs = ["add-256-4/add-256-4_direct.cpp"],
    hdrs = ["add-256-4/add-256-4_direct.h"],
    deps = [
        "@openfhe//:pke",
    ],
)


cc_library(
    name = "main_add-num-1_common",
    srcs = ["add-num-1/main_add-num-1_common.cpp"],
    hdrs = ["add-num-1/main_add-num-1_common.h"],
    deps = [
        ":params",
        "@openfhe//:pke",
    ],
)

cc_binary(
    name = "main_add-num-1_bisection",
    srcs = ["add-num-1/main_add-num-1_bisection.cpp"],
    deps = [
        ":main_add-num-1_common",
        ":add-num-1_bisection",
    ],
)

cc_binary(
    name = "main_add-num-1_closed",
    srcs = ["add-num-1/main_add-num-1_closed.cpp"],
    deps = [
        ":main_add-num-1_common",
        ":add-num-1_closed",
    ],
)

cc_binary(
    name = "main_add-num-1_direct",
    srcs = ["add-num-1/main_add-num-1_direct.cpp"],
    deps = [
        ":main_add-num-1_common",
        ":add-num-1_direct",
    ],
)

cc_library(
    name = "add-num-1_bisection",
    srcs = ["add-num-1/add-num-1_bisection.cpp"],
    hdrs = ["add-num-1/add-num-1_bisection.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "add-num-1_closed",
    srcs = ["add-num-1/add-num-1_closed.cpp"],
    hdrs = ["add-num-1/add-num-1_closed.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "add-num-1_direct",
    srcs = ["add-num-1/add-num-1_direct.cpp"],
    hdrs = ["add-num-1/add-num-1_direct.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "main_add-num-32_common",
    srcs = ["add-num-32/main_add-num-32_common.cpp"],
    hdrs = ["add-num-32/main_add-num-32_common.h"],
    deps = [
        ":params",
        "@openfhe//:pke",
    ],
)

cc_binary(
    name = "main_add-num-32_bisection",
    srcs = ["add-num-32/main_add-num-32_bisection.cpp"],
    deps = [
        ":main_add-num-32_common",
        ":add-num-32_bisection",
    ],
)

cc_binary(
    name = "main_add-num-32_closed",
    srcs = ["add-num-32/main_add-num-32_closed.cpp"],
    deps = [
        ":main_add-num-32_common",
        ":add-num-32_closed",
    ],
)

cc_binary(
    name = "main_add-num-32_direct",
    srcs = ["add-num-32/main_add-num-32_direct.cpp"],
    deps = [
        ":main_add-num-32_common",
        ":add-num-32_direct",
    ],
)

cc_library(
    name = "add-num-32_bisection",
    srcs = ["add-num-32/add-num-32_bisection.cpp"],
    hdrs = ["add-num-32/add-num-32_bisection.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "add-num-32_closed",
    srcs = ["add-num-32/add-num-32_closed.cpp"],
    hdrs = ["add-num-32/add-num-32_closed.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "add-num-32_direct",
    srcs = ["add-num-32/add-num-32_direct.cpp"],
    hdrs = ["add-num-32/add-num-32_direct.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "main_add-num-64_common",
    srcs = ["add-num-64/main_add-num-64_common.cpp"],
    hdrs = ["add-num-64/main_add-num-64_common.h"],
    deps = [
        ":params",
        "@openfhe//:pke",
    ],
)

cc_binary(
    name = "main_add-num-64_bisection",
    srcs = ["add-num-64/main_add-num-64_bisection.cpp"],
    deps = [
        ":main_add-num-64_common",
        ":add-num-64_bisection",
    ],
)

cc_binary(
    name = "main_add-num-64_closed",
    srcs = ["add-num-64/main_add-num-64_closed.cpp"],
    deps = [
        ":main_add-num-64_common",
        ":add-num-64_closed",
    ],
)

cc_binary(
    name = "main_add-num-64_direct",
    srcs = ["add-num-64/main_add-num-64_direct.cpp"],
    deps = [
        ":main_add-num-64_common",
        ":add-num-64_direct",
    ],
)

cc_library(
    name = "add-num-64_bisection",
    srcs = ["add-num-64/add-num-64_bisection.cpp"],
    hdrs = ["add-num-64/add-num-64_bisection.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "add-num-64_closed",
    srcs = ["add-num-64/add-num-64_closed.cpp"],
    hdrs = ["add-num-64/add-num-64_closed.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "add-num-64_direct",
    srcs = ["add-num-64/add-num-64_direct.cpp"],
    hdrs = ["add-num-64/add-num-64_direct.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "main_add-num-128_common",
    srcs = ["add-num-128/main_add-num-128_common.cpp"],
    hdrs = ["add-num-128/main_add-num-128_common.h"],
    deps = [
        ":params",
        "@openfhe//:pke",
    ],
)

cc_binary(
    name = "main_add-num-128_bisection",
    srcs = ["add-num-128/main_add-num-128_bisection.cpp"],
    deps = [
        ":main_add-num-128_common",
        ":add-num-128_bisection",
    ],
)

cc_binary(
    name = "main_add-num-128_closed",
    srcs = ["add-num-128/main_add-num-128_closed.cpp"],
    deps = [
        ":main_add-num-128_common",
        ":add-num-128_closed",
    ],
)

cc_binary(
    name = "main_add-num-128_direct",
    srcs = ["add-num-128/main_add-num-128_direct.cpp"],
    deps = [
        ":main_add-num-128_common",
        ":add-num-128_direct",
    ],
)

cc_library(
    name = "add-num-128_bisection",
    srcs = ["add-num-128/add-num-128_bisection.cpp"],
    hdrs = ["add-num-128/add-num-128_bisection.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "add-num-128_closed",
    srcs = ["add-num-128/add-num-128_closed.cpp"],
    hdrs = ["add-num-128/add-num-128_closed.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "add-num-128_direct",
    srcs = ["add-num-128/add-num-128_direct.cpp"],
    hdrs = ["add-num-128/add-num-128_direct.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "main_rotate-num-128_common",
    srcs = ["rotate-num-128/main_rotate-num-128_common.cpp"],
    hdrs = ["rotate-num-128/main_rotate-num-128_common.h"],
    deps = [
        ":params",
        "@openfhe//:pke",
    ],
)

cc_binary(
    name = "main_rotate-num-128_bisection",
    srcs = ["rotate-num-128/main_rotate-num-128_bisection.cpp"],
    deps = [
        ":main_rotate-num-128_common",
        ":rotate-num-128_bisection",
    ],
)

cc_binary(
    name = "main_rotate-num-128_closed",
    srcs = ["rotate-num-128/main_rotate-num-128_closed.cpp"],
    deps = [
        ":main_rotate-num-128_common",
        ":rotate-num-128_closed",
    ],
)

cc_binary(
    name = "main_rotate-num-128_direct",
    srcs = ["rotate-num-128/main_rotate-num-128_direct.cpp"],
    deps = [
        ":main_rotate-num-128_common",
        ":rotate-num-128_direct",
    ],
)

cc_library(
    name = "rotate-num-128_bisection",
    srcs = ["rotate-num-128/rotate-num-128_bisection.cpp"],
    hdrs = ["rotate-num-128/rotate-num-128_bisection.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "rotate-num-128_closed",
    srcs = ["rotate-num-128/rotate-num-128_closed.cpp"],
    hdrs = ["rotate-num-128/rotate-num-128_closed.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "rotate-num-128_direct",
    srcs = ["rotate-num-128/rotate-num-128_direct.cpp"],
    hdrs = ["rotate-num-128/rotate-num-128_direct.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "main_rotate-num-64_common",
    srcs = ["rotate-num-64/main_rotate-num-64_common.cpp"],
    hdrs = ["rotate-num-64/main_rotate-num-64_common.h"],
    deps = [
        ":params",
        "@openfhe//:pke",
    ],
)

cc_binary(
    name = "main_rotate-num-64_bisection",
    srcs = ["rotate-num-64/main_rotate-num-64_bisection.cpp"],
    deps = [
        ":main_rotate-num-64_common",
        ":rotate-num-64_bisection",
    ],
)

cc_binary(
    name = "main_rotate-num-64_closed",
    srcs = ["rotate-num-64/main_rotate-num-64_closed.cpp"],
    deps = [
        ":main_rotate-num-64_common",
        ":rotate-num-64_closed",
    ],
)

cc_binary(
    name = "main_rotate-num-64_direct",
    srcs = ["rotate-num-64/main_rotate-num-64_direct.cpp"],
    deps = [
        ":main_rotate-num-64_common",
        ":rotate-num-64_direct",
    ],
)

cc_library(
    name = "rotate-num-64_bisection",
    srcs = ["rotate-num-64/rotate-num-64_bisection.cpp"],
    hdrs = ["rotate-num-64/rotate-num-64_bisection.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "rotate-num-64_closed",
    srcs = ["rotate-num-64/rotate-num-64_closed.cpp"],
    hdrs = ["rotate-num-64/rotate-num-64_closed.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "rotate-num-64_direct",
    srcs = ["rotate-num-64/rotate-num-64_direct.cpp"],
    hdrs = ["rotate-num-64/rotate-num-64_direct.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "main_rotate-num-32_common",
    srcs = ["rotate-num-32/main_rotate-num-32_common.cpp"],
    hdrs = ["rotate-num-32/main_rotate-num-32_common.h"],
    deps = [
        ":params",
        "@openfhe//:pke",
    ],
)

cc_binary(
    name = "main_rotate-num-32_bisection",
    srcs = ["rotate-num-32/main_rotate-num-32_bisection.cpp"],
    deps = [
        ":main_rotate-num-32_common",
        ":rotate-num-32_bisection",
    ],
)

cc_binary(
    name = "main_rotate-num-32_closed",
    srcs = ["rotate-num-32/main_rotate-num-32_closed.cpp"],
    deps = [
        ":main_rotate-num-32_common",
        ":rotate-num-32_closed",
    ],
)

cc_binary(
    name = "main_rotate-num-32_direct",
    srcs = ["rotate-num-32/main_rotate-num-32_direct.cpp"],
    deps = [
        ":main_rotate-num-32_common",
        ":rotate-num-32_direct",
    ],
)

cc_library(
    name = "rotate-num-32_bisection",
    srcs = ["rotate-num-32/rotate-num-32_bisection.cpp"],
    hdrs = ["rotate-num-32/rotate-num-32_bisection.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "rotate-num-32_closed",
    srcs = ["rotate-num-32/rotate-num-32_closed.cpp"],
    hdrs = ["rotate-num-32/rotate-num-32_closed.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "rotate-num-32_direct",
    srcs = ["rotate-num-32/rotate-num-32_direct.cpp"],
    hdrs = ["rotate-num-32/rotate-num-32_direct.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "main_rotate-num-1_common",
    srcs = ["rotate-num-1/main_rotate-num-1_common.cpp"],
    hdrs = ["rotate-num-1/main_rotate-num-1_common.h"],
    deps = [
        ":params",
        "@openfhe//:pke",
    ],
)

cc_binary(
    name = "main_rotate-num-1_bisection",
    srcs = ["rotate-num-1/main_rotate-num-1_bisection.cpp"],
    deps = [
        ":main_rotate-num-1_common",
        ":rotate-num-1_bisection",
    ],
)

cc_binary(
    name = "main_rotate-num-1_closed",
    srcs = ["rotate-num-1/main_rotate-num-1_closed.cpp"],
    deps = [
        ":main_rotate-num-1_common",
        ":rotate-num-1_closed",
    ],
)

cc_binary(
    name = "main_rotate-num-1_direct",
    srcs = ["rotate-num-1/main_rotate-num-1_direct.cpp"],
    deps = [
        ":main_rotate-num-1_common",
        ":rotate-num-1_direct",
    ],
)

cc_library(
    name = "rotate-num-1_bisection",
    srcs = ["rotate-num-1/rotate-num-1_bisection.cpp"],
    hdrs = ["rotate-num-1/rotate-num-1_bisection.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "rotate-num-1_closed",
    srcs = ["rotate-num-1/rotate-num-1_closed.cpp"],
    hdrs = ["rotate-num-1/rotate-num-1_closed.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "rotate-num-1_direct",
    srcs = ["rotate-num-1/rotate-num-1_direct.cpp"],
    hdrs = ["rotate-num-1/rotate-num-1_direct.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "main_add-eq-7_common",
    srcs = ["add-eq-7/main_add-eq-7_common.cpp"],
    hdrs = ["add-eq-7/main_add-eq-7_common.h"],
    deps = [
        ":params",
        "@openfhe//:pke",
    ],
)

cc_binary(
    name = "main_add-eq-7_bisection",
    srcs = ["add-eq-7/main_add-eq-7_bisection.cpp"],
    deps = [
        ":main_add-eq-7_common",
        ":add-eq-7_bisection",
    ],
)

cc_binary(
    name = "main_add-eq-7_closed",
    srcs = ["add-eq-7/main_add-eq-7_closed.cpp"],
    deps = [
        ":main_add-eq-7_common",
        ":add-eq-7_closed",
    ],
)

cc_binary(
    name = "main_add-eq-7_direct",
    srcs = ["add-eq-7/main_add-eq-7_direct.cpp"],
    deps = [
        ":main_add-eq-7_common",
        ":add-eq-7_direct",
    ],
)

cc_library(
    name = "add-eq-7_bisection",
    srcs = ["add-eq-7/add-eq-7_bisection.cpp"],
    hdrs = ["add-eq-7/add-eq-7_bisection.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "add-eq-7_closed",
    srcs = ["add-eq-7/add-eq-7_closed.cpp"],
    hdrs = ["add-eq-7/add-eq-7_closed.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "add-eq-7_direct",
    srcs = ["add-eq-7/add-eq-7_direct.cpp"],
    hdrs = ["add-eq-7/add-eq-7_direct.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "main_add-eq-8_common",
    srcs = ["add-eq-8/main_add-eq-8_common.cpp"],
    hdrs = ["add-eq-8/main_add-eq-8_common.h"],
    deps = [
        ":params",
        "@openfhe//:pke",
    ],
)

cc_binary(
    name = "main_add-eq-8_bisection",
    srcs = ["add-eq-8/main_add-eq-8_bisection.cpp"],
    deps = [
        ":main_add-eq-8_common",
        ":add-eq-8_bisection",
    ],
)

cc_binary(
    name = "main_add-eq-8_closed",
    srcs = ["add-eq-8/main_add-eq-8_closed.cpp"],
    deps = [
        ":main_add-eq-8_common",
        ":add-eq-8_closed",
    ],
)

cc_binary(
    name = "main_add-eq-8_direct",
    srcs = ["add-eq-8/main_add-eq-8_direct.cpp"],
    deps = [
        ":main_add-eq-8_common",
        ":add-eq-8_direct",
    ],
)

cc_library(
    name = "add-eq-8_bisection",
    srcs = ["add-eq-8/add-eq-8_bisection.cpp"],
    hdrs = ["add-eq-8/add-eq-8_bisection.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "add-eq-8_closed",
    srcs = ["add-eq-8/add-eq-8_closed.cpp"],
    hdrs = ["add-eq-8/add-eq-8_closed.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "add-eq-8_direct",
    srcs = ["add-eq-8/add-eq-8_direct.cpp"],
    hdrs = ["add-eq-8/add-eq-8_direct.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "main_add-eq-10_common",
    srcs = ["add-eq-10/main_add-eq-10_common.cpp"],
    hdrs = ["add-eq-10/main_add-eq-10_common.h"],
    deps = [
        ":params",
        "@openfhe//:pke",
    ],
)

cc_binary(
    name = "main_add-eq-10_bisection",
    srcs = ["add-eq-10/main_add-eq-10_bisection.cpp"],
    deps = [
        ":main_add-eq-10_common",
        ":add-eq-10_bisection",
    ],
)

cc_binary(
    name = "main_add-eq-10_closed",
    srcs = ["add-eq-10/main_add-eq-10_closed.cpp"],
    deps = [
        ":main_add-eq-10_common",
        ":add-eq-10_closed",
    ],
)

cc_binary(
    name = "main_add-eq-10_direct",
    srcs = ["add-eq-10/main_add-eq-10_direct.cpp"],
    deps = [
        ":main_add-eq-10_common",
        ":add-eq-10_direct",
    ],
)

cc_library(
    name = "add-eq-10_bisection",
    srcs = ["add-eq-10/add-eq-10_bisection.cpp"],
    hdrs = ["add-eq-10/add-eq-10_bisection.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "add-eq-10_closed",
    srcs = ["add-eq-10/add-eq-10_closed.cpp"],
    hdrs = ["add-eq-10/add-eq-10_closed.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "add-eq-10_direct",
    srcs = ["add-eq-10/add-eq-10_direct.cpp"],
    hdrs = ["add-eq-10/add-eq-10_direct.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "main_add-eq-9_common",
    srcs = ["add-eq-9/main_add-eq-9_common.cpp"],
    hdrs = ["add-eq-9/main_add-eq-9_common.h"],
    deps = [
        ":params",
        "@openfhe//:pke",
    ],
)

cc_binary(
    name = "main_add-eq-9_bisection",
    srcs = ["add-eq-9/main_add-eq-9_bisection.cpp"],
    deps = [
        ":main_add-eq-9_common",
        ":add-eq-9_bisection",
    ],
)

cc_binary(
    name = "main_add-eq-9_closed",
    srcs = ["add-eq-9/main_add-eq-9_closed.cpp"],
    deps = [
        ":main_add-eq-9_common",
        ":add-eq-9_closed",
    ],
)

cc_binary(
    name = "main_add-eq-9_direct",
    srcs = ["add-eq-9/main_add-eq-9_direct.cpp"],
    deps = [
        ":main_add-eq-9_common",
        ":add-eq-9_direct",
    ],
)

cc_library(
    name = "add-eq-9_bisection",
    srcs = ["add-eq-9/add-eq-9_bisection.cpp"],
    hdrs = ["add-eq-9/add-eq-9_bisection.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "add-eq-9_closed",
    srcs = ["add-eq-9/add-eq-9_closed.cpp"],
    hdrs = ["add-eq-9/add-eq-9_closed.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "add-eq-9_direct",
    srcs = ["add-eq-9/add-eq-9_direct.cpp"],
    hdrs = ["add-eq-9/add-eq-9_direct.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "main_add-eq-11_common",
    srcs = ["add-eq-11/main_add-eq-11_common.cpp"],
    hdrs = ["add-eq-11/main_add-eq-11_common.h"],
    deps = [
        ":params",
        "@openfhe//:pke",
    ],
)

cc_binary(
    name = "main_add-eq-11_bisection",
    srcs = ["add-eq-11/main_add-eq-11_bisection.cpp"],
    deps = [
        ":main_add-eq-11_common",
        ":add-eq-11_bisection",
    ],
)

cc_binary(
    name = "main_add-eq-11_closed",
    srcs = ["add-eq-11/main_add-eq-11_closed.cpp"],
    deps = [
        ":main_add-eq-11_common",
        ":add-eq-11_closed",
    ],
)

cc_binary(
    name = "main_add-eq-11_direct",
    srcs = ["add-eq-11/main_add-eq-11_direct.cpp"],
    deps = [
        ":main_add-eq-11_common",
        ":add-eq-11_direct",
    ],
)

cc_library(
    name = "add-eq-11_bisection",
    srcs = ["add-eq-11/add-eq-11_bisection.cpp"],
    hdrs = ["add-eq-11/add-eq-11_bisection.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "add-eq-11_closed",
    srcs = ["add-eq-11/add-eq-11_closed.cpp"],
    hdrs = ["add-eq-11/add-eq-11_closed.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "add-eq-11_direct",
    srcs = ["add-eq-11/add-eq-11_direct.cpp"],
    hdrs = ["add-eq-11/add-eq-11_direct.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "main_add-eq-12_common",
    srcs = ["add-eq-12/main_add-eq-12_common.cpp"],
    hdrs = ["add-eq-12/main_add-eq-12_common.h"],
    deps = [
        ":params",
        "@openfhe//:pke",
    ],
)

cc_binary(
    name = "main_add-eq-12_bisection",
    srcs = ["add-eq-12/main_add-eq-12_bisection.cpp"],
    deps = [
        ":main_add-eq-12_common",
        ":add-eq-12_bisection",
    ],
)

cc_binary(
    name = "main_add-eq-12_closed",
    srcs = ["add-eq-12/main_add-eq-12_closed.cpp"],
    deps = [
        ":main_add-eq-12_common",
        ":add-eq-12_closed",
    ],
)

cc_binary(
    name = "main_add-eq-12_direct",
    srcs = ["add-eq-12/main_add-eq-12_direct.cpp"],
    deps = [
        ":main_add-eq-12_common",
        ":add-eq-12_direct",
    ],
)

cc_library(
    name = "add-eq-12_bisection",
    srcs = ["add-eq-12/add-eq-12_bisection.cpp"],
    hdrs = ["add-eq-12/add-eq-12_bisection.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "add-eq-12_closed",
    srcs = ["add-eq-12/add-eq-12_closed.cpp"],
    hdrs = ["add-eq-12/add-eq-12_closed.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "add-eq-12_direct",
    srcs = ["add-eq-12/add-eq-12_direct.cpp"],
    hdrs = ["add-eq-12/add-eq-12_direct.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "main_add-eq-13_common",
    srcs = ["add-eq-13/main_add-eq-13_common.cpp"],
    hdrs = ["add-eq-13/main_add-eq-13_common.h"],
    deps = [
        ":params",
        "@openfhe//:pke",
    ],
)

cc_binary(
    name = "main_add-eq-13_bisection",
    srcs = ["add-eq-13/main_add-eq-13_bisection.cpp"],
    deps = [
        ":main_add-eq-13_common",
        ":add-eq-13_bisection",
    ],
)

cc_binary(
    name = "main_add-eq-13_closed",
    srcs = ["add-eq-13/main_add-eq-13_closed.cpp"],
    deps = [
        ":main_add-eq-13_common",
        ":add-eq-13_closed",
    ],
)

cc_binary(
    name = "main_add-eq-13_direct",
    srcs = ["add-eq-13/main_add-eq-13_direct.cpp"],
    deps = [
        ":main_add-eq-13_common",
        ":add-eq-13_direct",
    ],
)

cc_library(
    name = "add-eq-13_bisection",
    srcs = ["add-eq-13/add-eq-13_bisection.cpp"],
    hdrs = ["add-eq-13/add-eq-13_bisection.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "add-eq-13_closed",
    srcs = ["add-eq-13/add-eq-13_closed.cpp"],
    hdrs = ["add-eq-13/add-eq-13_closed.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "add-eq-13_direct",
    srcs = ["add-eq-13/add-eq-13_direct.cpp"],
    hdrs = ["add-eq-13/add-eq-13_direct.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "main_add-eq-14_common",
    srcs = ["add-eq-14/main_add-eq-14_common.cpp"],
    hdrs = ["add-eq-14/main_add-eq-14_common.h"],
    deps = [
        ":params",
        "@openfhe//:pke",
    ],
)

cc_binary(
    name = "main_add-eq-14_bisection",
    srcs = ["add-eq-14/main_add-eq-14_bisection.cpp"],
    deps = [
        ":main_add-eq-14_common",
        ":add-eq-14_bisection",
    ],
)

cc_binary(
    name = "main_add-eq-14_closed",
    srcs = ["add-eq-14/main_add-eq-14_closed.cpp"],
    deps = [
        ":main_add-eq-14_common",
        ":add-eq-14_closed",
    ],
)

cc_binary(
    name = "main_add-eq-14_direct",
    srcs = ["add-eq-14/main_add-eq-14_direct.cpp"],
    deps = [
        ":main_add-eq-14_common",
        ":add-eq-14_direct",
    ],
)

cc_library(
    name = "add-eq-14_bisection",
    srcs = ["add-eq-14/add-eq-14_bisection.cpp"],
    hdrs = ["add-eq-14/add-eq-14_bisection.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "add-eq-14_closed",
    srcs = ["add-eq-14/add-eq-14_closed.cpp"],
    hdrs = ["add-eq-14/add-eq-14_closed.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "add-eq-14_direct",
    srcs = ["add-eq-14/add-eq-14_direct.cpp"],
    hdrs = ["add-eq-14/add-eq-14_direct.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "main_add-eq-16_common",
    srcs = ["add-eq-16/main_add-eq-16_common.cpp"],
    hdrs = ["add-eq-16/main_add-eq-16_common.h"],
    deps = [
        ":params",
        "@openfhe//:pke",
    ],
)

cc_binary(
    name = "main_add-eq-16_bisection",
    srcs = ["add-eq-16/main_add-eq-16_bisection.cpp"],
    deps = [
        ":main_add-eq-16_common",
        ":add-eq-16_bisection",
    ],
)

cc_binary(
    name = "main_add-eq-16_closed",
    srcs = ["add-eq-16/main_add-eq-16_closed.cpp"],
    deps = [
        ":main_add-eq-16_common",
        ":add-eq-16_closed",
    ],
)

cc_binary(
    name = "main_add-eq-16_direct",
    srcs = ["add-eq-16/main_add-eq-16_direct.cpp"],
    deps = [
        ":main_add-eq-16_common",
        ":add-eq-16_direct",
    ],
)

cc_library(
    name = "add-eq-16_bisection",
    srcs = ["add-eq-16/add-eq-16_bisection.cpp"],
    hdrs = ["add-eq-16/add-eq-16_bisection.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "add-eq-16_closed",
    srcs = ["add-eq-16/add-eq-16_closed.cpp"],
    hdrs = ["add-eq-16/add-eq-16_closed.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "add-eq-16_direct",
    srcs = ["add-eq-16/add-eq-16_direct.cpp"],
    hdrs = ["add-eq-16/add-eq-16_direct.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "main_add-eq-18_common",
    srcs = ["add-eq-18/main_add-eq-18_common.cpp"],
    hdrs = ["add-eq-18/main_add-eq-18_common.h"],
    deps = [
        ":params",
        "@openfhe//:pke",
    ],
)

cc_binary(
    name = "main_add-eq-18_bisection",
    srcs = ["add-eq-18/main_add-eq-18_bisection.cpp"],
    deps = [
        ":main_add-eq-18_common",
        ":add-eq-18_bisection",
    ],
)

cc_binary(
    name = "main_add-eq-18_closed",
    srcs = ["add-eq-18/main_add-eq-18_closed.cpp"],
    deps = [
        ":main_add-eq-18_common",
        ":add-eq-18_closed",
    ],
)

cc_binary(
    name = "main_add-eq-18_direct",
    srcs = ["add-eq-18/main_add-eq-18_direct.cpp"],
    deps = [
        ":main_add-eq-18_common",
        ":add-eq-18_direct",
    ],
)

cc_library(
    name = "add-eq-18_bisection",
    srcs = ["add-eq-18/add-eq-18_bisection.cpp"],
    hdrs = ["add-eq-18/add-eq-18_bisection.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "add-eq-18_closed",
    srcs = ["add-eq-18/add-eq-18_closed.cpp"],
    hdrs = ["add-eq-18/add-eq-18_closed.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "add-eq-18_direct",
    srcs = ["add-eq-18/add-eq-18_direct.cpp"],
    hdrs = ["add-eq-18/add-eq-18_direct.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "main_add-eq-20_common",
    srcs = ["add-eq-20/main_add-eq-20_common.cpp"],
    hdrs = ["add-eq-20/main_add-eq-20_common.h"],
    deps = [
        ":params",
        "@openfhe//:pke",
    ],
)

cc_binary(
    name = "main_add-eq-20_bisection",
    srcs = ["add-eq-20/main_add-eq-20_bisection.cpp"],
    deps = [
        ":main_add-eq-20_common",
        ":add-eq-20_bisection",
    ],
)

cc_binary(
    name = "main_add-eq-20_closed",
    srcs = ["add-eq-20/main_add-eq-20_closed.cpp"],
    deps = [
        ":main_add-eq-20_common",
        ":add-eq-20_closed",
    ],
)

cc_binary(
    name = "main_add-eq-20_direct",
    srcs = ["add-eq-20/main_add-eq-20_direct.cpp"],
    deps = [
        ":main_add-eq-20_common",
        ":add-eq-20_direct",
    ],
)

cc_library(
    name = "add-eq-20_bisection",
    srcs = ["add-eq-20/add-eq-20_bisection.cpp"],
    hdrs = ["add-eq-20/add-eq-20_bisection.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "add-eq-20_closed",
    srcs = ["add-eq-20/add-eq-20_closed.cpp"],
    hdrs = ["add-eq-20/add-eq-20_closed.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "add-eq-20_direct",
    srcs = ["add-eq-20/add-eq-20_direct.cpp"],
    hdrs = ["add-eq-20/add-eq-20_direct.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "main_rotate-of-1_common",
    srcs = ["rotate-of-1/main_rotate-of-1_common.cpp"],
    hdrs = ["rotate-of-1/main_rotate-of-1_common.h"],
    deps = [
        ":params",
        "@openfhe//:pke",
    ],
)

cc_binary(
    name = "main_rotate-of-1_bisection",
    srcs = ["rotate-of-1/main_rotate-of-1_bisection.cpp"],
    deps = [
        ":main_rotate-of-1_common",
        ":rotate-of-1_bisection",
    ],
)

cc_binary(
    name = "main_rotate-of-1_closed",
    srcs = ["rotate-of-1/main_rotate-of-1_closed.cpp"],
    deps = [
        ":main_rotate-of-1_common",
        ":rotate-of-1_closed",
    ],
)

cc_binary(
    name = "main_rotate-of-1_direct",
    srcs = ["rotate-of-1/main_rotate-of-1_direct.cpp"],
    deps = [
        ":main_rotate-of-1_common",
        ":rotate-of-1_direct",
    ],
)

cc_library(
    name = "rotate-of-1_bisection",
    srcs = ["rotate-of-1/rotate-of-1_bisection.cpp"],
    hdrs = ["rotate-of-1/rotate-of-1_bisection.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "rotate-of-1_closed",
    srcs = ["rotate-of-1/rotate-of-1_closed.cpp"],
    hdrs = ["rotate-of-1/rotate-of-1_closed.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "rotate-of-1_direct",
    srcs = ["rotate-of-1/rotate-of-1_direct.cpp"],
    hdrs = ["rotate-of-1/rotate-of-1_direct.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "main_rotate-of-2_common",
    srcs = ["rotate-of-2/main_rotate-of-2_common.cpp"],
    hdrs = ["rotate-of-2/main_rotate-of-2_common.h"],
    deps = [
        ":params",
        "@openfhe//:pke",
    ],
)

cc_binary(
    name = "main_rotate-of-2_bisection",
    srcs = ["rotate-of-2/main_rotate-of-2_bisection.cpp"],
    deps = [
        ":main_rotate-of-2_common",
        ":rotate-of-2_bisection",
    ],
)

cc_binary(
    name = "main_rotate-of-2_closed",
    srcs = ["rotate-of-2/main_rotate-of-2_closed.cpp"],
    deps = [
        ":main_rotate-of-2_common",
        ":rotate-of-2_closed",
    ],
)

cc_binary(
    name = "main_rotate-of-2_direct",
    srcs = ["rotate-of-2/main_rotate-of-2_direct.cpp"],
    deps = [
        ":main_rotate-of-2_common",
        ":rotate-of-2_direct",
    ],
)

cc_library(
    name = "rotate-of-2_bisection",
    srcs = ["rotate-of-2/rotate-of-2_bisection.cpp"],
    hdrs = ["rotate-of-2/rotate-of-2_bisection.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "rotate-of-2_closed",
    srcs = ["rotate-of-2/rotate-of-2_closed.cpp"],
    hdrs = ["rotate-of-2/rotate-of-2_closed.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "rotate-of-2_direct",
    srcs = ["rotate-of-2/rotate-of-2_direct.cpp"],
    hdrs = ["rotate-of-2/rotate-of-2_direct.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "main_rotate-of-3_common",
    srcs = ["rotate-of-3/main_rotate-of-3_common.cpp"],
    hdrs = ["rotate-of-3/main_rotate-of-3_common.h"],
    deps = [
        ":params",
        "@openfhe//:pke",
    ],
)

cc_binary(
    name = "main_rotate-of-3_bisection",
    srcs = ["rotate-of-3/main_rotate-of-3_bisection.cpp"],
    deps = [
        ":main_rotate-of-3_common",
        ":rotate-of-3_bisection",
    ],
)

cc_binary(
    name = "main_rotate-of-3_closed",
    srcs = ["rotate-of-3/main_rotate-of-3_closed.cpp"],
    deps = [
        ":main_rotate-of-3_common",
        ":rotate-of-3_closed",
    ],
)

cc_binary(
    name = "main_rotate-of-3_direct",
    srcs = ["rotate-of-3/main_rotate-of-3_direct.cpp"],
    deps = [
        ":main_rotate-of-3_common",
        ":rotate-of-3_direct",
    ],
)

cc_library(
    name = "rotate-of-3_bisection",
    srcs = ["rotate-of-3/rotate-of-3_bisection.cpp"],
    hdrs = ["rotate-of-3/rotate-of-3_bisection.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "rotate-of-3_closed",
    srcs = ["rotate-of-3/rotate-of-3_closed.cpp"],
    hdrs = ["rotate-of-3/rotate-of-3_closed.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "rotate-of-3_direct",
    srcs = ["rotate-of-3/rotate-of-3_direct.cpp"],
    hdrs = ["rotate-of-3/rotate-of-3_direct.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "main_rotate-of-4_common",
    srcs = ["rotate-of-4/main_rotate-of-4_common.cpp"],
    hdrs = ["rotate-of-4/main_rotate-of-4_common.h"],
    deps = [
        ":params",
        "@openfhe//:pke",
    ],
)

cc_binary(
    name = "main_rotate-of-4_bisection",
    srcs = ["rotate-of-4/main_rotate-of-4_bisection.cpp"],
    deps = [
        ":main_rotate-of-4_common",
        ":rotate-of-4_bisection",
    ],
)

cc_binary(
    name = "main_rotate-of-4_closed",
    srcs = ["rotate-of-4/main_rotate-of-4_closed.cpp"],
    deps = [
        ":main_rotate-of-4_common",
        ":rotate-of-4_closed",
    ],
)

cc_binary(
    name = "main_rotate-of-4_direct",
    srcs = ["rotate-of-4/main_rotate-of-4_direct.cpp"],
    deps = [
        ":main_rotate-of-4_common",
        ":rotate-of-4_direct",
    ],
)

cc_library(
    name = "rotate-of-4_bisection",
    srcs = ["rotate-of-4/rotate-of-4_bisection.cpp"],
    hdrs = ["rotate-of-4/rotate-of-4_bisection.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "rotate-of-4_closed",
    srcs = ["rotate-of-4/rotate-of-4_closed.cpp"],
    hdrs = ["rotate-of-4/rotate-of-4_closed.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "rotate-of-4_direct",
    srcs = ["rotate-of-4/rotate-of-4_direct.cpp"],
    hdrs = ["rotate-of-4/rotate-of-4_direct.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "main_add-large-0_common",
    srcs = ["add-large-0/main_add-large-0_common.cpp"],
    hdrs = ["add-large-0/main_add-large-0_common.h"],
    deps = [
        ":params",
        "@openfhe//:pke",
    ],
)

cc_binary(
    name = "main_add-large-0_bisection",
    srcs = ["add-large-0/main_add-large-0_bisection.cpp"],
    deps = [
        ":main_add-large-0_common",
        ":add-large-0_bisection",
    ],
)

cc_binary(
    name = "main_add-large-0_closed",
    srcs = ["add-large-0/main_add-large-0_closed.cpp"],
    deps = [
        ":main_add-large-0_common",
        ":add-large-0_closed",
    ],
)

cc_binary(
    name = "main_add-large-0_direct",
    srcs = ["add-large-0/main_add-large-0_direct.cpp"],
    deps = [
        ":main_add-large-0_common",
        ":add-large-0_direct",
    ],
)

cc_library(
    name = "add-large-0_bisection",
    srcs = ["add-large-0/add-large-0_bisection.cpp"],
    hdrs = ["add-large-0/add-large-0_bisection.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "add-large-0_closed",
    srcs = ["add-large-0/add-large-0_closed.cpp"],
    hdrs = ["add-large-0/add-large-0_closed.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "add-large-0_direct",
    srcs = ["add-large-0/add-large-0_direct.cpp"],
    hdrs = ["add-large-0/add-large-0_direct.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "main_add-large-5_common",
    srcs = ["add-large-5/main_add-large-5_common.cpp"],
    hdrs = ["add-large-5/main_add-large-5_common.h"],
    deps = [
        ":params",
        "@openfhe//:pke",
    ],
)

cc_binary(
    name = "main_add-large-5_bisection",
    srcs = ["add-large-5/main_add-large-5_bisection.cpp"],
    deps = [
        ":main_add-large-5_common",
        ":add-large-5_bisection",
    ],
)

cc_binary(
    name = "main_add-large-5_closed",
    srcs = ["add-large-5/main_add-large-5_closed.cpp"],
    deps = [
        ":main_add-large-5_common",
        ":add-large-5_closed",
    ],
)

cc_binary(
    name = "main_add-large-5_direct",
    srcs = ["add-large-5/main_add-large-5_direct.cpp"],
    deps = [
        ":main_add-large-5_common",
        ":add-large-5_direct",
    ],
)

cc_library(
    name = "add-large-5_bisection",
    srcs = ["add-large-5/add-large-5_bisection.cpp"],
    hdrs = ["add-large-5/add-large-5_bisection.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "add-large-5_closed",
    srcs = ["add-large-5/add-large-5_closed.cpp"],
    hdrs = ["add-large-5/add-large-5_closed.h"],
    deps = [
        "@openfhe//:pke",
    ],
)

cc_library(
    name = "add-large-5_direct",
    srcs = ["add-large-5/add-large-5_direct.cpp"],
    hdrs = ["add-large-5/add-large-5_direct.h"],
    deps = [
        "@openfhe//:pke",
    ],
)
