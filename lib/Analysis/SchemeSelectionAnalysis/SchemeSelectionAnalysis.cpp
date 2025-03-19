#include "lib/Analysis/SchemeSelectionAnalysis/SchemeSelectionAnalysis.h"

#include <algorithm>
#include <cassert>
#include <functional>

#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Analysis/Utils.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"              // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"      // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"        // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"                 // from @llvm-project
#include "mlir/include/mlir/Interfaces/CallInterfaces.h"   // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project

namespace mlir {
namespace heir {

LogicalResult SchemeSelectionAnalysis::visitOperation(
    Operation *op, ArrayRef<const SchemeInfoLattice *> operands,
    ArrayRef<SchemeInfoLattice *> results) {
  auto propagate = [&](Value value, const NatureOfComputation &counter) {
    auto *lattice = getLatticeElement(value);
    ChangeResult changed = lattice->join(counter);
    propagateIfChanged(lattice, changed);
  };

  llvm::TypeSwitch<Operation &>(*op)
      .Case<secret::GenericOp>([&](auto genericOp) {
        Block *body = genericOp.getBody();
        for (auto i = 0; i != body->getNumArguments(); ++i) {
          auto blockArg = body->getArgument(i);
          propagate(blockArg, NatureOfComputation());
        }
      })
      .Case<arith::AddIOp, arith::SubIOp>([&](auto addOp) {
        SmallVector<OpResult> secretResults;
        getSecretResults(op, secretResults);
        if (secretResults.empty()) {
          return;
        }

        // TODO: switch for real and ints
        NatureOfComputation intArithOpCount(0, 0, 1, 0, 0, 0);
        propagate(addOp->getResult(0), intArithOpCount);
      });
  return success();
}

bool hasAtLeastOneBooleanOperand(Operation *op) {
  for (Value operand : op->getOperands()) {
    // Check if the operand type is a Boolean type
    if (operand.getType().isInteger(1)) {
      return true;
    }
  }
  return false;
}

bool hasAtLeastOneIntegerOperand(Operation *op) {
  for (Value operand : op->getOperands()) {
    // Check if the operand type is an integer type
    // note that this also includes Boolean type i1
    if (operand.getType().isInteger()) {
      return true;
    }
  }
  return false;
}

bool hasAtLeastOneRealOperand(Operation *op) {
  for (Value operand : op->getOperands()) {
    // Check if the operand type is a floating-point type
    if (operand.getType().isF16() || operand.getType().isF32() ||
        operand.getType().isF64() || operand.getType().isF128()) {
      return true;
    }
  }
  return false;
}

void annotateNatureOfComputation(Operation *top, DataFlowSolver *solver,
                                 int baseLevel) {
  top->walk<WalkOrder::PreOrder>([&](secret::GenericOp genericOp) {
    genericOp.getBody()->walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (op->getNumResults() == 0) {
        return;
      }
      if (!isSecret(op->getResult(0), solver)) {
        return;
      }

      // Create a SmallVector to hold the components of natOfComp
      SmallVector<Attribute, 4> natOfCompValues;

      // Check each condition and add to the array if true
      if (hasAtLeastOneBooleanOperand(op)) {
        natOfCompValues.push_back(StringAttr::get(top->getContext(), "bool"));
      }
      if (hasAtLeastOneIntegerOperand(op)) {
        natOfCompValues.push_back(StringAttr::get(top->getContext(), "int"));
      }
      if (hasAtLeastOneRealOperand(op)) {
        natOfCompValues.push_back(StringAttr::get(top->getContext(), "real"));
      }
      // Add checks for other conditions as needed
      // e.g., hasAtLeastOneBitOperand(op), hasAtLeastOneComparisonOperand(op),
      // etc.

      // If any natOfComp values were collected, set the attribute
      if (!natOfCompValues.empty()) {
        auto natOfCompAttr = ArrayAttr::get(top->getContext(), natOfCompValues);
        op->setAttr("natOfComp", natOfCompAttr);
      }
    });
  });
}

}  // namespace heir
}  // namespace mlir
