load("@heir//lib/Transforms:transforms.bzl", "add_heir_transforms")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "AnnotateParameters",
    srcs = ["AnnotateParameters.cpp"],
    hdrs = [
        "AnnotateParameters.h",
    ],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Analysis/OperationCountAnalysis",
        "@heir//lib/Dialect/Mgmt/IR:Dialect",
        "@heir//lib/Dialect/BGV/IR:Dialect",
        "@heir//lib/Dialect/Secret/IR:Dialect",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
    ],
)

add_heir_transforms(
    generated_target_name = "pass_inc_gen",
    pass_name = "AnnotateParameters",
)
