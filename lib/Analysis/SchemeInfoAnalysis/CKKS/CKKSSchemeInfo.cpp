// CKKSSchemeInfoPass.cpp

#include "CKKSSchemeInfo.h"

#include <algorithm>
#include <cassert>
#include <functional>
#include <unordered_map>

#include "CKKSSchemeInfo.h"

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

#define DEBUG_TYPE "CKKSSchemeInfo"

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

    // TODO: Adjust runtime
    static const std::unordered_map<std::string, std::vector<int> > opRuntimeMap = {
        {"arith.addf", {2, 4, 6, 8}},
        {"arith.mulf", {127, 322, 339, 447}},
        {"mgmt.relinearize", {0, 1, 2, 2}},
        {"mgmt.rescale", {0, 1, 2, 2}},
        {"tensor_ext.rotate", {88, 196, 208, 291}}
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

    LogicalResult CKKSSchemeInfoAnalysis::visitOperation(
        Operation *op, ArrayRef<const CKKSSchemeInfoLattice *> operands,
        ArrayRef<CKKSSchemeInfoLattice *> results) {
        LLVM_DEBUG(llvm::dbgs() << "Visiting: " << op->getName() << ". ");

        auto propagate = [&](Value value, const CKKSSchemeInfo &info) {
            auto *oldInfo = getLatticeElement(value);
            ChangeResult changed = oldInfo->join(info);
            propagateIfChanged(oldInfo, changed);
        };

        auto getOperandLevel = [&]() {
            int level = 0;
            for (auto const schemeInfoLattice: operands) {
                level = std::max(schemeInfoLattice->getValue().getLevel(), level);
            }
            return level;
        };

        auto propagateLevelToResult = [&](bool increase = false) {
            auto level = getOperandLevel();
            level = increase ? level + 1 : level;
            propagate(op->getResult(0), CKKSSchemeInfo(level));
        };

        llvm::TypeSwitch<Operation &>(*op)
            // count floating-point arithmetic ops
            .Case<arith::AddFOp>([&](auto op) {
                propagateLevelToResult();
            })
            .Case<arith::MulFOp>([&](auto op) {
                // CKKS: multiplication increases multiplicative depth; rescale modeled in runtime only.
                propagateLevelToResult(true);
            })
            .Case<linalg::MatvecOp>([&](linalg::MatvecOp matvecOp) {
                // Only the main iteration is considered; one mult depth overall.
                propagateLevelToResult(true);
            })
            .Case<affine::AffineForOp>([&](affine::AffineForOp forOp) {
                // Base level entering the loop.
                int baseLevel = getOperandLevel();

                auto iterCountOpt = affine::getConstantTripCount(forOp);
                if (!iterCountOpt.has_value()) {
                    throw std::runtime_error("Expected constant trip count for affine.for");
                }
                auto iterCount = iterCountOpt.value();

                llvm::DenseMap<Value, int> depth;

                Block *body = forOp.getBody();
                for (BlockArgument arg: body->getArguments()) {
                    depth[arg] = 0;
                    propagate(arg, CKKSSchemeInfo(baseLevel));
                }

                forOp.getRegion().walk<WalkOrder::PreOrder>([&](Operation *op) {
                    int inDepth = 0;
                    for (Value v: op->getOperands()) {
                        auto it = depth.find(v);
                        if (it != depth.end())
                            inDepth = std::max(inDepth, it->second);
                    }

                    int outDepth = (isa<arith::MulFOp>(op) || isa<linalg::MatvecOp>(op))
                                       ? inDepth + 1
                                       : inDepth;

                    for (Value r: op->getResults()) {
                        auto it = depth.find(r);
                        if (it == depth.end())
                            depth[r] = outDepth;
                        else
                            it->second = std::max(it->second, outDepth);
                    }
                });

                // Propagate levels for all values based on their depth
                for (auto const &entry: depth) {
                    Value value = entry.first;
                    int valueDepth = entry.second;
                    propagate(value, CKKSSchemeInfo(baseLevel + valueDepth));
                }

                // Find the terminator and compute the maximum depth contributing to iter args.
                auto yieldOp = dyn_cast<affine::AffineYieldOp>(body->getTerminator());

                int bodyIncrease = 0;
                for (Value v: yieldOp.getOperands()) {
                    auto it = depth.find(v);
                    if (it != depth.end())
                        bodyIncrease = std::max(bodyIncrease, it->second);
                }

                // Extra depth from repeated iterations.
                int extra = static_cast<int>(iterCount * bodyIncrease);

                // Apply the extra depth to the loop's yielded results (iter args).
                for (Value res: forOp.getResults()) {
                    propagate(res, CKKSSchemeInfo(baseLevel + extra));
                }
            });

        return success();
    }

    // Helper: compute the operand level for an operation from the solver state.
    // Uses max level among initialized operands; defaults to 1 if unknown/no operands.
    static int getOperandLevel(Operation *op, DataFlowSolver *solver, int maxLevel) {
        int level = 0;
        for (auto operand: op->getOperands()) {
            if (auto *state = solver->lookupState<CKKSSchemeInfoLattice>(operand)) {
                auto val = state->getValue();
                if (val.isInitialized()) {
                    level = std::min(level, val.getLevel());
                }
            }
        }
        return maxLevel - level;
    }

    static int getMaxLevel(Operation *top, DataFlowSolver *solver) {
        auto maxLevel = 0;
        walkValues(top, [&](Value value) {
            auto levelState = solver->lookupState<CKKSSchemeInfoLattice>(value)->getValue();
            if (levelState.isInitialized()) {
                auto level = levelState.getLevel();
                maxLevel = std::max(maxLevel, level);
            }
        });
        return maxLevel;
    }

    static int computeRuntimeForRegion(Region &region, DataFlowSolver *solver, int maxLevel, int levelOffset = 0) {
        int runtime = 0;
        region.walk<WalkOrder::PreOrder>([&](Operation *top) {
            int level = getOperandLevel(top, solver, maxLevel) - levelOffset;

            llvm::TypeSwitch<Operation &>(*top)
                .Case<arith::AddFOp>([&](auto op) {
                    runtime += getRuntime(op.getOperation(), level);
                })
                .Case<arith::MulFOp>([&](auto op) {
                    runtime += getRuntime(op.getOperation(), level);
                    runtime += getRuntime("mgmt.relinearize", level);
                    runtime += getRuntime("mgmt.rescale", level);
                })
                .Case<affine::AffineForOp>([&](affine::AffineForOp forOp) {
                    auto tripCountOpt = affine::getConstantTripCount(forOp);
                    if (!tripCountOpt.has_value()) {
                        return;
                    }
                    auto tripCount = tripCountOpt.value();
                    for (unsigned i = 0; i < tripCount; i++) {
                        runtime += computeRuntimeForRegion(forOp.getRegion(), solver, maxLevel,
                                                           levelOffset + i);
                    }
                })
                .Case<linalg::MatvecOp>([&](linalg::MatvecOp matvecOp) {
                    auto inputs = matvecOp.getInputs();
                    auto matrixType = dyn_cast<RankedTensorType>(inputs[0].getType());
                    auto rows = matrixType.getShape()[0];
                    int lvl = getOperandLevel(matvecOp.getOperation(), solver, maxLevel);

                    // R rotates, R multiplies (with relinearize + rescale per mul), and R adds.
                    runtime += rows * getRuntime("tensor_ext.rotate", lvl);
                    runtime += rows * getRuntime("arith.mulf", lvl);
                    runtime += rows * getRuntime("mgmt.relinearize", lvl);
                    runtime += rows * getRuntime("mgmt.rescale", lvl);
                    runtime += rows * getRuntime("arith.addf", lvl);
                })
                .Default([&](auto &op) {
                    if (isIllegalDialect(&op) || isIllegalOperation(&op)) {
                        op.emitError(
                            "Unsupported Operation for CKKS scheme. Cannot provide runtime estimation for scheme selection");
                    }
                    LLVM_DEBUG(
                        llvm::dbgs() << "Unsupported Operation for CKKS runtime estimation " << op.getName() <<
                        "\n");
                });
        });

        return runtime;
    }

    int computeApproximateRuntimeCKKS(Operation *top, DataFlowSolver *solver) {
        // maxLevel is currently unused but retained for potential future logic.
        auto maxLevel = getMaxLevel(top, solver);

        int runtime = 0;
        top->walk<WalkOrder::PreOrder>([&](func::FuncOp funcOp) {
            runtime = computeRuntimeForRegion(funcOp.getBody(), solver, maxLevel);
        });
        return runtime;
    }

} // namespace heir
} // namespace mlir