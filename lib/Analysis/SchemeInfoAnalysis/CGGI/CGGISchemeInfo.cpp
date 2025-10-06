#include "CGGISchemeInfo.h"

#include <algorithm>
#include <cassert>
#include <functional>
#include <unordered_map>

#include "CGGISchemeInfo.h"

#include <mlir/Dialect/Affine/Analysis/LoopAnalysis.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h.inc>
#include "lib/Analysis/Utils.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Utils/Utils.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"              // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"               // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"     // from @llvm-project
#include "mlir/include/mlir/Dialect/Math/IR/Math.h"        // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"        // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"                 // from @llvm-project
#include "mlir/include/mlir/Interfaces/CallInterfaces.h"   // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project

#define DEBUG_TYPE "CGGISchemeInfo"

namespace mlir {
namespace heir {

static const std::unordered_map<std::string, int> opRuntimeMap = {
  {"arith.addi", 100},
  {"arith.subi", 100},
  {"arith.muli", 400},
  {"mgmt.relinearize", 200},
  {"mgmt.modreduce", 300},
  {"tensor_ext.rotate", 400},
};

static int getRuntime(const std::string &opName) {
  LLVM_DEBUG(llvm::dbgs() << "Looking up: " << opName << "\n");
  auto const entry = opRuntimeMap.find(opName);
  LLVM_DEBUG(llvm::dbgs() << "Found: " << entry->second << "\n");
  if (entry != opRuntimeMap.end()) {
    return entry->second;
  }
  return -1;
};

static int getRuntime(Operation *op) {
  auto const opName = op->getName().getStringRef().lower();
  return getRuntime(opName);
};

LogicalResult CGGISchemeInfoAnalysis::visitOperation(
    Operation *op, ArrayRef<const CGGISchemeInfoLattice *> operands,
    ArrayRef<CGGISchemeInfoLattice *> results) {
  LLVM_DEBUG(llvm::dbgs() << "Visiting: " << op->getName() << ". ");

  auto propagate = [&](Value value, const CGGISchemeInfo &info) {
    auto *oldInfo = getLatticeElement(value);
    ChangeResult changed = oldInfo->join(info);
    propagateIfChanged(oldInfo, changed);
  };

  llvm::TypeSwitch<Operation &>(*op)
      // count integer arithmetic ops
      .Case<arith::AddIOp, arith::SubIOp>([&](auto op) {

      })
      .Case<arith::MulIOp>([&](auto op) {

      })
  	  .Case<affine::AffineForOp>([&](affine::AffineForOp forOp) {
      // TODO: Implement
  	  });
  return success();
}

static int computeRuntimeForRegion(Region &region) {
  int runtime = 0;
  auto addRuntime = [&runtime](int additional) {
    runtime += additional;
  };

  region.walk<WalkOrder::PreOrder>([&](Operation *top) {
     llvm::TypeSwitch<Operation &>(*top)
      // count integer arithmetic ops
      .Case<arith::AddIOp, arith::SubIOp, arith::MulIOp>([&](auto op) {
        addRuntime(getRuntime(op));
      })
     .Case<affine::AffineForOp>([&](affine::AffineForOp forOp) {
       auto tripCountOpt = affine::getConstantTripCount(forOp);
       if (!tripCountOpt.has_value()) {
         return;
       }
       auto tripCount = tripCountOpt.value();
       auto roundTime = computeRuntimeForRegion(forOp.getRegion());
       addRuntime(tripCount * roundTime);
     })

      .Default([&](auto& op) {
        LLVM_DEBUG(llvm::dbgs() << "Unsupported Operation for CGGI runtime estimation " << op.getName() << "\n");
      });
   });

  return runtime;
}

int computeApproximateRuntimeCGGI(Operation *top, DataFlowSolver *solver) {
  int runtime = 0;

  top->walk<WalkOrder::PreOrder>([&](func::FuncOp funcOp) {
    runtime = computeRuntimeForRegion(funcOp.getBody());
  });

  return runtime;
}


}  // namespace heir
}  // namespace mlir
