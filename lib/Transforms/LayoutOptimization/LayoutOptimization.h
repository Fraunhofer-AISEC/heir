#ifndef LIB_TRANSFORMS_LAYOUTOPTIMIZATION_LAYOUTOPTIMIZATION_H_
#define LIB_TRANSFORMS_LAYOUTOPTIMIZATION_LAYOUTOPTIMIZATION_H_

#include "lib/Dialect/Secret/IR/SecretDialect.h"
#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/Pass/Pass.h"                 // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DECL
#include "lib/Transforms/LayoutOptimization/LayoutOptimization.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Transforms/LayoutOptimization/LayoutOptimization.h.inc"

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TRANSFORMS_LAYOUTOPTIMIZATION_LAYOUTOPTIMIZATION_H_
