# OperationCountAnalysis analysis pass
package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "OperationCountAnalysis",
    srcs = ["OperationCountAnalysis.cpp"],
    hdrs = ["OperationCountAnalysis.h"],
    deps = [
        "@heir//lib/Dialect:Utils",
        "@heir//lib/Dialect/Openfhe/IR:Dialect",
        "@heir//lib/Dialect/Secret/IR:Dialect",
        "@heir//lib/Dialect/TensorExt/IR:Dialect",
        "@heir//lib/Dialect/Mgmt/IR:Dialect",
        "@heir//lib/Analysis/SecretnessAnalysis",
        "@heir//lib/Analysis/LevelAnalysis",
        "@openfhe//:core",
        "@openfhe//:pke",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)