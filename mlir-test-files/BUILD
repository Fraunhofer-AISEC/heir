cc_library(
    name = "dot8",
    srcs = ["dot8/dot8.cpp"],
    hdrs = ["dot8/dot8.h"],
    deps = [
        "@openfhe//:pke"
    ],
)

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