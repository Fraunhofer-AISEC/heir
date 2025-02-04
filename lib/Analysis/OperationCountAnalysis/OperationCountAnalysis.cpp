#include "lib/Analysis/OperationCountAnalysis/OperationCountAnalysis.h"
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include <cassert>
#include <cstdint>

#include "OperationCountAnalysis.h"
#include "lib/Analysis/LevelAnalysis/LevelAnalysis.h"
#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/Mgmt/IR/MgmtOps.h"
#include "lib/Dialect/Mgmt/IR/MgmtAttributes.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "src/core/include/math/hal/vector.h"                //from @openfhe
#include "src/core/include/math/nbtheory.h"                //from @openfhe
#include "src/core/include/math/hal/nativeintbackend.h"        //from @openfhe
#include "llvm/include/llvm/ADT/TypeSwitch.h"              // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"               // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"      // from @llvm-project
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
        auto secretness = isSecretInternal(op, op->getOperand(0));
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
        auto secretness = isSecretInternal(extractOp, extractOp->getOperand(0));
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

static uint64_t computeModulusOrder(int ringDimension, uint64_t plaintextModulus) {
  uint64_t cyclOrder = 2 * ringDimension;
  uint64_t pow2ptm = 1;

  while (plaintextModulus % 2 == 0) {
    plaintextModulus >>= 1;
    pow2ptm <<= 1;
  }

  if (pow2ptm < cyclOrder) {
    pow2ptm = cyclOrder;
  }

  return pow2ptm * plaintextModulus;
}

static uint64_t findValidFirstModSize(int minModSize, int ringDimension, int plaintextModulus) {
  uint64_t modulusOrder = computeModulusOrder(ringDimension, plaintextModulus);

  while (minModSize < 60) {
    try {
      lbcrypto::LastPrime<lbcrypto::NativeInteger>(minModSize, modulusOrder);
      return minModSize;
    } catch (lbcrypto::OpenFHEException &e) {
      minModSize += 1;
    }
  }
  return 0;
}

static uint64_t findValidScalingModSize(int minModSize, int firstModSize, int numPrimes, int ringDimension, int plaintextModulus) {
  uint64_t modulusOrder = computeModulusOrder(ringDimension, plaintextModulus);

  lbcrypto::NativeInteger firstModulus = 0;
  if (firstModSize < minModSize) {
    firstModulus = lbcrypto::LastPrime<lbcrypto::NativeInteger>(firstModSize, modulusOrder);
  }

  while (minModSize < 60) {
    try {
      auto q = lbcrypto::LastPrime<lbcrypto::NativeInteger>(minModSize, modulusOrder);
      for (int i = 1; i < numPrimes; i++) {
        q = lbcrypto::PreviousPrime<lbcrypto::NativeInteger>(q, modulusOrder);
        if (q == firstModulus) {
          minModSize += 1;
          continue;
        }
      }
      return minModSize;
    } catch (lbcrypto::OpenFHEException &e) {
      minModSize += 1;
    }
  }
  return 0;
}

void annotateCountParams(Operation *top, DataFlowSolver *solver,
                         int ringDimension, int plaintextModulus) {
  top->walk<WalkOrder::PreOrder>([&](secret::GenericOp genericOp) {
    int maxKeySwitchCount = 0;
    int maxCiphertextCount = 0;

    genericOp.getBody()->walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (op->getNumResults() == 0) {
        return;
      }
      auto *lattice = solver->lookupState<OperationCountLattice>(op->getResult(0));
      if (!lattice) {
        return;
      }

      auto count = lattice->getValue();
      if (!count.isInitialized()) {
        return;
      }

      maxKeySwitchCount = std::max(maxKeySwitchCount, count.getKeySwitchCount());
      maxCiphertextCount = std::max(maxCiphertextCount, count.getCiphertextCount());
    });

    auto log2 = [](double x) { return log(x) / log(2); };

    auto maxLevel = 0;
    top->walk<WalkOrder::PreOrder>([&](secret::GenericOp genericOp) {
      genericOp.getBody()->walk<WalkOrder::PreOrder>([&](Operation *op) {
        if (op->getNumResults() == 0) {
          return;
        }
        if (!isSecret(op->getResult(0), solver)) {
          return;
        }
        // ensure result is secret
        auto level = solver->lookupState<LevelLattice>(op->getResult(0))
                        ->getValue().getLevel();
        maxLevel = std::max(maxLevel, level);
      });
    });

    auto multiplicativeDepth = maxLevel;
    auto numPrimes = multiplicativeDepth + 1;

    // Compute param sizes for HYBRID Key Switching
    auto phi = ringDimension;
    auto beta = pow(2.0, 10); // TODO Replace by value set through developer
    auto t = plaintextModulus;
    auto D = 6.0;

    auto vKey = 2.0 / 3.0; // TODO Make adjustable
    auto vErr = 3.19 * 3.19; // TODO Make adjustable

    auto boundScale = D * t * sqrt((phi / 12.0) * (1.0 + (phi * vKey)));

    auto boundKeySwitch = D * t * phi * sqrt(vErr / 12.0);

    auto K = 100.0;
    auto P = K * beta * sqrt(log(numPrimes * pow(2.0, 60)) / log(beta));

    auto f0 = beta * sqrt(numPrimes * log2(t * phi)) / P;

    auto vKS = f0 * boundKeySwitch + boundScale;

    auto bOptimal = boundScale + sqrt(boundScale * boundScale + (maxKeySwitchCount * vKS));
    auto pOptimal = 2.0 * maxCiphertextCount * bOptimal;

    auto scalingModSize = ceil(log2(pOptimal));
    if (scalingModSize > 60) {
      top->emitOpError() << "ScalingModSize too large (> 60 bit).\n";
    }

    double firstModSize = ceil(1 + log2( bOptimal));
    firstModSize = findValidFirstModSize(firstModSize, ringDimension, plaintextModulus); 
    if (!firstModSize) {
      top->emitOpError() << "Cannot find valid prime for scalingModSize\n";
    };
    

    scalingModSize = findValidScalingModSize(scalingModSize,firstModSize , numPrimes, ringDimension, plaintextModulus);
    if (!scalingModSize) {
      top->emitOpError() << "Cannot find enough valid primes for scalingModSize\n";
    };

    // annotate mgmt::OpenfheParamsAttr to func::FuncOp containing the genericOp
    auto *funcOp = genericOp->getParentOp();
    auto openfheParamAttr = mgmt::OpenfheParamsAttr::get(
        funcOp->getContext(), multiplicativeDepth, scalingModSize, firstModSize);
    funcOp->setAttr(mgmt::MgmtDialect::kArgOpenfheParamsAttrName, openfheParamAttr);
  });
}

}  // namespace heir
}  // namespace mlir