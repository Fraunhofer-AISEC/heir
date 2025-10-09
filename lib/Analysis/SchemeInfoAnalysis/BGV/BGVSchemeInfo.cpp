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

// Runtime map: rounded milliseconds (int), indexed by operand level (1-4)
static const std::unordered_map<std::string, std::vector<int>> opRuntimeMap = {
  {"arith.addi",        {2,   4,   6,   8}},   // levels 1-4
  {"arith.subi",        {2,   4,   6,   8}},   // levels 1-4
  {"arith.muli",        {127, 322, 339, 447}}, // levels 1-4
  {"mgmt.relinearize",  {0,   1,   2,   2}},   // levels 1-4
  {"mgmt.modreduce",    {0,   1,   2,   2}},   // levels 1-4
  {"tensor_ext.rotate", {88,  196, 208, 291}}, // levels 1-4
};

static int getRuntime(const std::string &opName, int level) {
  LLVM_DEBUG(llvm::dbgs() << "Looking up: " << opName << " at level " << level << "\n");
  auto const entry = opRuntimeMap.find(opName);
  if (entry != opRuntimeMap.end()) {
    const auto &timings = entry->second;
    // Clamp level to valid range [1-4], use 0-based indexing
    int idx = std::max(0, std::min(3, level - 1));
    int runtime = timings[idx];
    LLVM_DEBUG(llvm::dbgs() << "Found: " << runtime << " ms\n");
    return runtime;
  }
  LLVM_DEBUG(llvm::dbgs() << "Operation not found in runtime map\n");
  return -1;
}

static int getRuntime(Operation *op, int level) {
  auto const opName = op->getName().getStringRef().lower();
  return getRuntime(opName, level);
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

// Helper: compute the operand level for an operation from the solver state.
// Uses max level among initialized operands; defaults to 1 if unknown/no operands.
static int getOperandLevel(Operation *op, DataFlowSolver *solver, int maxLevel) {
  int level = 0;
  for (auto operand : op->getOperands()) {
    if (auto *state = solver->lookupState<BGVSchemeInfoLattice>(operand)) {
      auto val = state->getValue();
      if (val.isInitialized()) {
        level = std::min(level, val.getLevel());
      }
    }
  }
  return maxLevel - level;
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

static int computeRuntimeForRegion(Region &region, DataFlowSolver *solver, int maxLevel) {
  int runtime = 0;
  auto addRuntime = [&runtime](int additional) {
    runtime += additional;
  };

  region.walk<WalkOrder::PreOrder>([&](Operation *top) {
     int level = getOperandLevel(top, solver, maxLevel);

     llvm::TypeSwitch<Operation &>(*top)
      .Case<arith::AddIOp, arith::SubIOp>([&](auto op) {
        addRuntime(getRuntime(op.getOperation(), level));
      })
      .Case<arith::MulIOp>([&](auto op) {
        addRuntime(getRuntime(op.getOperation(), level));
        addRuntime(getRuntime("mgmt.relinearize", level));
        addRuntime(getRuntime("mgmt.modreduce", level));
      })
     .Case<affine::AffineForOp>([&](affine::AffineForOp forOp) {
       auto tripCountOpt = affine::getConstantTripCount(forOp);
       if (!tripCountOpt.has_value()) {
         return;
       }
       auto tripCount = tripCountOpt.value();
       auto roundTime = computeRuntimeForRegion(forOp.getRegion(), solver, maxLevel);
       addRuntime(tripCount * roundTime);
     })
    .Case<linalg::MatvecOp>([&](linalg::MatvecOp matvecOp) {
        auto inputs = matvecOp.getInputs();
        auto matrixType = dyn_cast<RankedTensorType>(inputs[0].getType());
        auto rows = matrixType.getShape()[0];
        int lvl = getOperandLevel(matvecOp.getOperation(), solver, maxLevel);

        // R rotates, R multiplies (with relinearize + modreduce per mul), and R adds.
        addRuntime(rows * getRuntime("tensor_ext.rotate", lvl));
        addRuntime(rows * getRuntime("arith.muli", lvl));
        addRuntime(rows * getRuntime("mgmt.relinearize", lvl));
        addRuntime(rows * getRuntime("mgmt.modreduce", lvl));
        addRuntime(rows * getRuntime("arith.addi", lvl));
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
  // maxLevel is currently unused but retained for potential future logic.
  auto maxLevel = getMaxLevel(top, solver);

  int runtime = 0;
  top->walk<WalkOrder::PreOrder>([&](func::FuncOp funcOp) {
    runtime = computeRuntimeForRegion(funcOp.getBody(), solver, maxLevel);
  });
  return runtime;
}

}  // namespace heir
}  // namespace mlir