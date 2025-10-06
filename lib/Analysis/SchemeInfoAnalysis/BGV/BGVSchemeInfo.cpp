#include "BGVSchemeInfo.h"

#include <algorithm>
#include <cassert>
#include <functional>
#include <unordered_map>

#include "BGVSchemeInfo.h"

#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Utils/Utils.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"              // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"               // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h" // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"     // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"    // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"        // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"                 // from @llvm-project

#define DEBUG_TYPE "BGVSchemeInfo"

namespace mlir {
namespace heir {

static const std::vector<std::string> illegalDialects = {
  "comb"
};

static const std::vector<std::string> illegalOperations = {
  "arith.cmpi",
  "arith.xori",
  // ...
};

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

static bool isIllegalDialect(Operation *op) {
  std::string dialectName = op->getName().getDialect()->getNamespace().str();
  return std::find(illegalDialects.begin(), illegalDialects.end(), dialectName) != illegalDialects.end();
}

static bool isIllegalOperation(Operation *op) {
  std::string opName = op->getName().getStringRef().lower();
  return std::find(illegalOperations.begin(), illegalOperations.end(), opName) != illegalOperations.end();
}

LogicalResult BGVSchemeInfoAnalysis::visitOperation(
    Operation *op, ArrayRef<const BGVSchemeInfoLattice *> operands,
    ArrayRef<BGVSchemeInfoLattice *> results) {
  LLVM_DEBUG(llvm::dbgs() << "Visiting: " << op->getName() << ". ");

  auto propagate = [&](Value value, const BGVSchemeInfo &info) {
    auto *oldInfo = getLatticeElement(value);
    ChangeResult changed = oldInfo->join(info);
    propagateIfChanged(oldInfo, changed);
  };

  auto getOperandLevel = [&]() {
    int level = 0;
    for (auto const schemeInfoLattice : operands) {
      level = std::max(schemeInfoLattice->getValue().getLevel(), level);
    }
    return level;
  };

  auto propagateLevelToResult = [&](bool increase = false) {
    auto level = getOperandLevel();
    level = increase ? level + 1 : level;
    propagate(op->getResult(0), BGVSchemeInfo(level));
  };

  llvm::TypeSwitch<Operation &>(*op)
      // count integer arithmetic ops
      .Case<arith::AddIOp, arith::SubIOp>([&](auto op) {
        propagateLevelToResult();
      })
      .Case<arith::MulIOp>([&](auto op) {
        propagateLevelToResult(true);
      })
      .Case<linalg::MatvecOp>([&](linalg::MatvecOp matvecOp) {
        // Only the main iteration is considered; one mult depth overall.
        propagateLevelToResult(true);
      })
      .Case<affine::AffineForOp>([&](affine::AffineForOp forOp) {
         // TODO: Implement
      });

  return success();
}

static int getMaxLevel(Operation* top, DataFlowSolver* solver) {
  auto maxLevel = 0;
  walkValues(top, [&](Value value) {
      auto levelState = solver->lookupState<BGVSchemeInfoLattice>(value)->getValue();
      if (levelState.isInitialized()) {
        auto level = levelState.getLevel();
        maxLevel = std::max(maxLevel, level);
      }
    });
  return maxLevel;
}


static int computeRuntimeForRegion(Region &region) {
  int runtime = 0;
  auto addRuntime = [&runtime](int additional) {
    runtime += additional;
  };

  region.walk<WalkOrder::PreOrder>([&](Operation *top) {
     llvm::TypeSwitch<Operation &>(*top)
      // count integer arithmetic ops
      .Case<arith::AddIOp, arith::SubIOp>([&](auto op) {
        addRuntime(getRuntime(op));
      })
      .Case<arith::MulIOp>([&](auto op) {
        addRuntime(getRuntime(op));
        addRuntime(getRuntime("mgmt.relinearize"));
        addRuntime(getRuntime("mgmt.modreduce"));
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
    .Case<linalg::MatvecOp>([&](linalg::MatvecOp matvecOp) {
        auto inputs = matvecOp.getInputs();
        auto matrixType = dyn_cast<RankedTensorType>(inputs[0].getType());
        auto rows = matrixType.getShape()[0];

        // R multiplies (with relinearize + modreduce per mul) and R adds.
        addRuntime(rows * getRuntime("tensor_ext.rotate"));
        addRuntime(rows * getRuntime("arith.muli"));
        addRuntime(rows * getRuntime("mgmt.relinearize"));
        addRuntime(rows * getRuntime("mgmt.modreduce"));
        addRuntime(rows * getRuntime("arith.addi"));
      })
    .Default([&](auto& op) {
      if (isIllegalDialect(&op) || isIllegalOperation(&op)) {
        op.emitError("Unsupported Operation for BGV scheme. Cannot provide runtime estimation for scheme selection");
      }
      LLVM_DEBUG(llvm::dbgs() << "Unsupported Operation for BGV runtime estimation " << op.getName() << "\n");
    });
   });

  return runtime;
}

int computeApproximateRuntimeBGV(Operation *top, DataFlowSolver *solver) {
  int maxLevel = getMaxLevel(top, solver);

  auto getLevel = [&](Value value) {
    auto levelState = solver->lookupState<BGVSchemeInfoLattice>(value)->getValue();
    if (levelState.isInitialized()) {
      return maxLevel - levelState.getLevel();
    }
    return -1;
  };

  auto getOperationLevel = [&](Operation *op) {
    int level = maxLevel;
    for (auto operand : op->getOperands()) {
      level = std::min(level, getLevel(operand));
    }
    return level;
  };

  int runtime = 0;
  top->walk<WalkOrder::PreOrder>([&](func::FuncOp funcOp) {
    runtime = computeRuntimeForRegion(funcOp.getBody());
  });
  return runtime;
}


}  // namespace heir
}  // namespace mlir
