#include "lib/Analysis/OperationCountAnalysis/OperationCountAnalysis.h"
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include <cassert>

#include "lib/Dialect/Mgmt/IR/MgmtOps.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"              // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"               // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"    // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"                 // from @llvm-project

#define DEBUG_TYPE "openfhe-operation-count-analysis"

namespace mlir {
namespace heir {

LogicalResult OperationCountAnalysis::visitOperation(
    Operation *op, 
    ArrayRef<const OperationCountLattice *> operands,
    ArrayRef<OperationCountLattice *> results) {

  auto propagate = [&](Value value, const OperationCount &counter) {
    auto *lattice = getLatticeElement(value);
    ChangeResult result = lattice->join(counter);

    propagateIfChanged(lattice, result);
  };

  llvm::TypeSwitch<Operation *>(op)
      .Case<secret::GenericOp>([&](auto genericOp){
        Block *body = genericOp.getBody();
        for (auto arg: body->getArguments()) {
          propagate(arg, OperationCount(1, 0));
        }
      })
      .Case<arith::AddIOp, arith::SubIOp>([&](auto addOp) {
        SmallVector<OpResult> secretResults;
        getSecretResults(op, secretResults);
        if (secretResults.empty()) {
         return;
        }

        OperationCount sumCount(0, 0);
        SmallVector<OpOperand *> secretOperands;
        getSecretOperands(op, secretOperands);
        for (auto *operand : secretOperands) {
          auto operationCount = operands[operand->getOperandNumber()]->getValue();
          sumCount = sumCount + operationCount;
        }
    
        propagate(addOp->getResult(0), sumCount);
      })
      .Case<arith::MulIOp>([&](auto &mulOp) {
        SmallVector<OpResult> secretResults;
        getSecretResults(op, secretResults);
        if (secretResults.empty()) {
          return;
        }
        propagate(mulOp->getResult(0), OperationCount(1, 0));
      })
      .Case<mgmt::RelinearizeOp, tensor_ext::RotateOp>([&](auto &op) {
        auto secretness = ensureSecretness(op, op->getOperand(0));
        if (!secretness) {
          return;
        }

        auto count = operands[0]->getValue();
        if (!count.isInitialized()) {
          return;
        }

        propagate(op.getResult(), count.incrementKeySwitch());
      })
      .Case<tensor::ExtractOp>([&] (auto &extractOp) {
        auto secretness = ensureSecretness(extractOp, extractOp->getOperand(0));
        if (!secretness) {
          return;
        }
        // See issue #1174
        propagate(extractOp.getResult(), OperationCount(1, 1));  
      })
      .Case<mgmt::ModReduceOp>([&] (auto &modReduceOp) {
        propagate(modReduceOp.getResult(), OperationCount(0, 0));
      });

      return success();

  return mlir::success();
}

}  // namespace heir
}  // namespace mlir