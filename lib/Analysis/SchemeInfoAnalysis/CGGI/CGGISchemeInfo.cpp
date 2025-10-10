#include "CGGISchemeInfo.h"

#include <algorithm>
#include <cassert>
#include <functional>
#include <unordered_map>

#include <mlir/Dialect/Affine/Analysis/LoopAnalysis.h>
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

// Keep these empty unless future policy needs them.
static const std::vector<std::string> illegalDialects = {};
static const std::vector<std::string> illegalOperations = {};

// TFHE integer op timings (rounded to int ms) by bit-width.
// Only the operations listed in the tables are kept.
static const std::unordered_map<std::string, std::unordered_map<int, int>> tfheRuntimeMap = {
    {"arith.addi",
     {
         {8, 181},
         {16, 205},
         {32, 314},
         {64, 412},
         {128, 855},
     }},
    {"arith.subi",
     {
         {8, 183},
         {16, 213},
         {32, 324},  // 323.559 -> 324
         {64, 402},  // 401.691 -> 402
         {128, 832},
     }},
    {"arith.muli",
     {
         {8, 346},
         {16, 626},
         {32, 1421},
         {64, 4370},
         {128, 16428},
     }},
};

// Lookup runtime for a given operation name and bit-width. Returns -1 if unknown.
static int getRuntime(const std::string &opName, int bitWidth) {
  LLVM_DEBUG(llvm::dbgs() << "Looking up op: " << opName << " at bitWidth: " << bitWidth << "\n");
  auto opIt = tfheRuntimeMap.find(opName);
  if (opIt == tfheRuntimeMap.end()) {
    LLVM_DEBUG(llvm::dbgs() << "No entry for op: " << opName << "\n");
    return -1;
  }
  const auto &bwMap = opIt->second;
  auto bwIt = bwMap.find(bitWidth);
  if (bwIt == bwMap.end()) {
    LLVM_DEBUG(llvm::dbgs() << "No timing for bitWidth: " << bitWidth << " (op: " << opName << ")\n");
    return -1;
  }
  LLVM_DEBUG(llvm::dbgs() << "Found timing: " << bwIt->second << " ms\n");
  return bwIt->second;
}

static std::string getOpNameLower(Operation *op) {
  // StringRef::lower returns std::string; do not call .str() on it.
  return op->getName().getStringRef().lower();
}

static bool isIllegalDialect(Operation *op) {
  std::string dialectName = op->getName().getDialect()->getNamespace().str();
  return std::find(illegalDialects.begin(), illegalDialects.end(), dialectName) != illegalDialects.end();
}

static bool isIllegalOperation(Operation *op) {
  std::string opName = op->getName().getStringRef().lower();
  return std::find(illegalOperations.begin(), illegalOperations.end(), opName) != illegalOperations.end();
}

static int getMaxIntegerBitWidthInOperands(Operation *op) {
  int bitWidth = -1;
  for (Value v : op->getOperands()) {
    if (auto intTy = dyn_cast<IntegerType>(v.getType())) {
      bitWidth = std::max(bitWidth, static_cast<int>(intTy.getWidth()));
    }
  }
  return bitWidth;
}

static int getFuncIntegerBitWidth(func::FuncOp funcOp) {
  // Inspect all integer inputs; if multiple bit-widths are present, use the maximum.
  int bitWidth = -1;
  for (Type t : funcOp.getFunctionType().getInputs()) {
    if (auto intTy = dyn_cast<IntegerType>(t)) {
      bitWidth = std::max(bitWidth, static_cast<int>(intTy.getWidth()));
    }
  }
  LLVM_DEBUG(llvm::dbgs() << "Func " << funcOp.getName() << " canonical integer bitWidth: " << bitWidth << "\n");
  return bitWidth;
}

LogicalResult CGGISchemeInfoAnalysis::visitOperation(
    Operation *op, ArrayRef<const CGGISchemeInfoLattice *> operands,
    ArrayRef<CGGISchemeInfoLattice *> results) {
  LLVM_DEBUG(llvm::dbgs() << "Visiting: " << op->getName() << ". ");

  llvm::TypeSwitch<Operation &>(*op)
      // Only the listed integer arithmetic ops are considered.
      .Case<arith::AddIOp, arith::SubIOp>([&](auto op) {
        (void)op;
      })
      .Case<arith::MulIOp>([&](auto op) {
        (void)op;
      })
      .Case<affine::AffineForOp>([&](affine::AffineForOp forOp) {
        (void)forOp;
      });
  return success();
}

// Compute runtime for a region given a chosen bit-width (from func inputs).
// If bitWidthFromFunc < 0, fallback to per-op operand integer bit-widths.
static int computeRuntimeForRegion(Region &region, int bitWidthFromFunc) {
  int runtime = 0;

  region.walk<WalkOrder::PreOrder>([&](Operation *top) {
    llvm::TypeSwitch<Operation &>(*top)
        .Case<arith::AddIOp, arith::SubIOp, arith::MulIOp>([&](auto op) {
          std::string opName = getOpNameLower(op.getOperation());
          int bw = bitWidthFromFunc;
          if (bw < 0) {
            bw = getMaxIntegerBitWidthInOperands(op.getOperation());
          }
          int rt = getRuntime(opName, bw);
          if (rt < 0) {
            LLVM_DEBUG(llvm::dbgs() << "Unsupported bitWidth (" << bw << ") for op " << opName << "\n");
            return;
          }
          runtime += rt;
        })
        .Case<affine::AffineForOp>([&](affine::AffineForOp forOp) {
          auto tripCountOpt = affine::getConstantTripCount(forOp);
          if (!tripCountOpt.has_value()) {
            // Non-constant loop: conservatively skip or could be handled via estimates.
            return;
          }
          auto tripCount = tripCountOpt.value();
          auto roundTime = computeRuntimeForRegion(forOp.getRegion(), bitWidthFromFunc);
          runtime += static_cast<int>(tripCount * roundTime);
        })
        .Default([&](auto &op) {
          // Operations not listed are ignored for timing; emit debug for visibility.
          if (isIllegalDialect(&op) || isIllegalOperation(&op)) {
            op.emitError("Unsupported Operation for CGGI scheme. Cannot provide runtime estimation for scheme selection");
          }
          LLVM_DEBUG(llvm::dbgs() << "Ignoring op for CGGI runtime estimation: " << op.getName() << "\n");
        });
  });

  return runtime;
}

int computeApproximateRuntimeCGGI(Operation *top, DataFlowSolver *solver) {
  (void)solver; // Unused in this computation.
  int totalRuntime = 0;

  top->walk<WalkOrder::PreOrder>([&](func::FuncOp funcOp) {
    int funcBitWidth = getFuncIntegerBitWidth(funcOp);
    int funcRuntime = computeRuntimeForRegion(funcOp.getBody(), funcBitWidth);
    totalRuntime += funcRuntime;
  });

  return totalRuntime;
}

}  // namespace heir
}  // namespace mlir