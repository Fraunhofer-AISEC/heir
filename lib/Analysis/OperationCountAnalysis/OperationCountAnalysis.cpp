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


// May be changed for the BGV GHS Variant
uint64_t findValidFirstModSize(int modSize, int rindDimension, int plaintextModulus) {
  // Schameless copy from OpenFHE
  uint32_t cyclOrder = 2 * rindDimension;
  uint32_t ptm = static_cast<uint32_t>(plaintextModulus);
  uint32_t pow2ptm = 1;  // The largest power of 2 dividing ptm (check
                      // whether it is larger than cyclOrder or not)
  while (ptm % 2 == 0) {
    ptm >>= 1;
    pow2ptm <<= 1;
  }

  if (pow2ptm < cyclOrder) {
    pow2ptm = cyclOrder;
  }

  uint64_t modulusOrder = pow2ptm * ptm;

  while (modSize < 60) {
    try {
      lbcrypto::LastPrime<lbcrypto::NativeInteger>(modSize, modulusOrder);
    } catch (lbcrypto::OpenFHEException &e) {
      modSize += 1;
    }
    return modSize;
  }
  return 0;
}

uint64_t findValidScalingModSize(int modSize, int numPrimes, int ringDimension, int plaintextModulus) {
  // Schameless copy from OpenFHE
  uint32_t cyclOrder = 2 * ringDimension;
  uint32_t ptm = static_cast<uint32_t>(plaintextModulus);
  uint32_t pow2ptm = 1;  // The largest power of 2 dividing ptm (check
                         // whether it is larger than cyclOrder or not)
  std::cerr << "CyclOrder: " << cyclOrder << "\n";
  std::cerr << "ptm: " << ptm << "\n";
  std::cerr << "modSize: " << modSize << "\n";
  std::cerr << "numPrimes: " << numPrimes << "\n";

  while (ptm % 2 == 0) {
    ptm >>= 1;
    pow2ptm <<= 1;
  }

  if (pow2ptm < cyclOrder) {
    pow2ptm = cyclOrder;
  }

  uint64_t modulusOrder = pow2ptm * ptm;

  while (modSize < 60) {
    try {
      auto q = lbcrypto::LastPrime<lbcrypto::NativeInteger>(modSize, modulusOrder);
      for (int i = 0; i < numPrimes; i++) {
        q = lbcrypto::PreviousPrime<lbcrypto::NativeInteger>(q, modulusOrder);
      }
      return modSize;
    } catch (lbcrypto::OpenFHEException &e) {
      modSize += 1;
    }
  }

  return 0;
}

void annotateCountParams(Operation *top, DataFlowSolver *solver,
                         int ringDimension, int plaintextModulus) {
  top->walk<WalkOrder::PreOrder>([&](secret::GenericOp genericOp) {
    int maxKeySwitchCount = 0;
    int maxAddCount = 0;

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
      maxAddCount = std::max(maxAddCount, count.getAddCount());
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
                        ->getValue()
                        .getLevel();
        maxLevel = std::max(maxLevel, level);
      });
    });

    auto nRotate = maxKeySwitchCount - 1; // TODO Clear up the OperationCount
    auto nAdd = maxAddCount;  

    // Compute param sizes for HYBRID Key Switching
    auto phi = ringDimension;
    auto beta = pow(2.0, 10);
    auto t = 2.0 * plaintextModulus;

    auto D = 6.0;

    auto vKey = 2.0 / 3.0;
    auto vErr = 3.19 * 3.19; // TODO Make adjustable

    // BScale and BClean
    auto bScale = D * t * sqrt((phi / 12.0) * (1.0 + (phi * vKey)));
    // BKS
    auto bKS = D * t * phi * sqrt(vErr / 12.0);

    auto K = 100.0;
    auto P = K * beta * sqrt(log((maxLevel + 1) * pow(2.0, 60)) / log(beta));

    auto f0 = beta * sqrt((maxLevel + 1) * log2(t * phi)) / P;
    auto f1 = 1.0;

    auto vKS = f0 * bKS + f1 * bScale;

    auto bOptimal = bScale + sqrt(bScale * bScale + ((nRotate + 1) * vKS));
    auto pOptimal = 2.0 * (nAdd + 1)  * bOptimal;

    std::cerr << "q (p*): " << pOptimal << "\n";

    std::cerr << "qi_size: " << pOptimal << "\n";
    std::cerr << "B*: " << log2(bOptimal) << "\n";

    std::cerr << "BScale: " << log2(bScale) << "\n";

    // Compute qi_size
    auto scalingModSize = ceil(log2(pOptimal));

    if (scalingModSize > 60) {
      top->emitOpError() << "q size too large (> 60 bit).\n";
    }

    scalingModSize = findValidScalingModSize(scalingModSize, maxLevel + 1, ringDimension, plaintextModulus);
    if (!scalingModSize) {
      top->emitOpError() << "Cannot find valid scalingModSize \n";
    };

    // Compute the first modulus (the last to be reduced)
    double firstModSize = ceil(log2(2 * bOptimal));

    firstModSize = findValidFirstModSize(firstModSize, ringDimension, plaintextModulus); 
    if (!firstModSize) {
      top->emitOpError() << "Cannot find valid firstModSize\n";
    };

    std::cerr << "ScalingModSize: " << scalingModSize << "\n";
    std::cerr << "FirstModSize: " << firstModSize << "\n";

    // annotate mgmt::OpenfheParamsAttr to func::FuncOp containing the genericOp
    auto *funcOp = genericOp->getParentOp();
    auto openfheParamAttr = mgmt::OpenfheParamsAttr::get(
        funcOp->getContext(), maxLevel + 1, scalingModSize, firstModSize);
    funcOp->setAttr(mgmt::MgmtDialect::kArgOpenfheParamsAttrName, openfheParamAttr);
  });
}

}  // namespace heir
}  // namespace mlir