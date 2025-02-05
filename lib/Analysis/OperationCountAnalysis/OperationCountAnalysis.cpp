#include "lib/Analysis/OperationCountAnalysis/OperationCountAnalysis.h"
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <iostream>

#include "OperationCountAnalysis.h"
#include "lib/Analysis/LevelAnalysis/LevelAnalysis.h"
#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/Mgmt/IR/MgmtOps.h"
#include "lib/Dialect/Mgmt/IR/MgmtAttributes.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "src/core/include/math/hal/vector.h"                //from @openfhe
#include "src/core/include/math/nbtheory.h"                //from @openfhe
#include "src/core/include/lattice/stdlatticeparms.h"       //from @openfhe
#include "src/pke/include/scheme/scheme-utils.h"            //from @openfhe
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

constexpr int MAX_BIT_SIZE = 60;

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

  while (minModSize < MAX_BIT_SIZE) {
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

  while (minModSize < MAX_BIT_SIZE) {
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

static int computeOptimalBoundSize(int ringDimension, int plaintextModulus, int maxKeySwitchCount, int maxCiphertextCount, int numPrimes) {
      auto phi = ringDimension;
      auto beta = pow(2.0, 10); // TODO Replace by value set through developer
      auto t = plaintextModulus;
      auto D = 6.0;

      auto vKey = 2.0 / 3.0; // TODO Make adjustable
      auto vErr = 3.19 * 3.19; // TODO Make adjustable

      auto boundScale = D * t * sqrt((phi / 12.0) * (1.0 + (phi * vKey)));

      auto boundKeySwitch = D * t * phi * sqrt(vErr / 12.0);

      auto K = 100.0;
      auto P = K * beta * sqrt(log(numPrimes * pow(2.0, MAX_BIT_SIZE)) / log(beta));

      auto f0 = beta * sqrt(numPrimes * log2(t * phi)) / P;

      auto vKS = f0 * boundKeySwitch + boundScale;

      return log2(boundScale + sqrt(boundScale * boundScale + (maxKeySwitchCount * vKS)));
}

void annotateCountParams(Operation *top, DataFlowSolver *solver,
                         int ringDimension, int plaintextModulus) {
  top->walk<WalkOrder::PreOrder>([&](secret::GenericOp genericOp) {
    int maxKeySwitchCount = 0;
    int maxCiphertextCount = 0;

    bool isRingDimensionSet = ringDimension != 0;

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

    uint32_t numPartQ = ComputeNumLargeDigits(0, multiplicativeDepth);
    auto auxBits = MAX_BIT_SIZE;  // Max size of a prime number in OpenFHE
    
    auto computeLogPQ = [&](int scalingModSize, int firstModSize, int numPrimes) {
      auto logQ = firstModSize + (numPrimes + 1) * scalingModSize;
      auto logP = ceil(ceil(static_cast<double>(logQ)  / numPartQ) / auxBits) * auxBits;

      return logP + logQ;
    };

    auto upperBoundOnModuli = lbcrypto::GetMSB(plaintextModulus) + 28;
    auto logQP = computeLogPQ(upperBoundOnModuli, upperBoundOnModuli, numPrimes);
    
    if (!isRingDimensionSet){
      ringDimension = 16384;
    }

    auto firstModSize = 0;
    auto scalingModSize = 0;
    
    while (true) {
      auto newRingDimension = ringDimension;
      auto startDimension = ringDimension;
      std::cerr << "Testing Ring Dimension: " << ringDimension << std::endl;
      // Compute param sizes for HYBRID Key Switching
      auto boundOptimal = computeOptimalBoundSize(startDimension, plaintextModulus, maxKeySwitchCount, maxCiphertextCount, numPrimes);
      auto moduliOptimal = ceil(1 + log2(maxCiphertextCount) + boundOptimal);

      scalingModSize = ceil(log2(moduliOptimal));
      if (scalingModSize > MAX_BIT_SIZE) {
        top->emitOpError() << "ScalingModSize too large (> 60 bit).\n";
      }

      firstModSize = ceil(1 + boundOptimal);
      firstModSize = findValidFirstModSize(firstModSize, startDimension, plaintextModulus); 
      if (!firstModSize) {
        top->emitOpError() << "Cannot find valid prime for scalingModSize\n";
      };

      scalingModSize = findValidScalingModSize(scalingModSize, firstModSize, numPrimes, startDimension, plaintextModulus);
      if (!scalingModSize) {
        top->emitOpError() << "Cannot find enough valid primes for scalingModSize\n";
      };

      logQP = computeLogPQ(scalingModSize, firstModSize, numPrimes);
      
      if (isRingDimensionSet) {
        break;
      }
      
      newRingDimension = lbcrypto::StdLatticeParm::FindRingDim(lbcrypto::HEStd_ternary, lbcrypto::HEStd_128_classic, logQP);

      if (newRingDimension != startDimension) {
        std::cerr << "Set new RingDimension: " << newRingDimension << std::endl;
        startDimension = newRingDimension;
      } else {
        startDimension /= 2;
        std::cerr << "Try smaller RingDimension: " << startDimension << std::endl;
        // Try smaller ring dimension


        // Compute param sizes for HYBRID Key Switching
        auto boundOptimal = computeOptimalBoundSize(startDimension, plaintextModulus, maxKeySwitchCount, maxCiphertextCount, numPrimes);
        auto moduliOptimal = ceil(1 + log2(maxCiphertextCount) + boundOptimal);

        auto testScalingModSize = ceil(log2(moduliOptimal));
        if (scalingModSize > MAX_BIT_SIZE) {
          top->emitOpError() << "ScalingModSize too large (> 60 bit).\n";
        }

        auto testFirstModSize = ceil(1 + boundOptimal);
        testFirstModSize = findValidFirstModSize(testFirstModSize, startDimension, plaintextModulus); 
        if (!testFirstModSize) {
          top->emitOpError() << "Cannot find valid prime for scalingModSize\n";
        };

        testScalingModSize = findValidScalingModSize(testScalingModSize, testFirstModSize, numPrimes, startDimension, plaintextModulus);
        if (!testScalingModSize) {
          top->emitOpError() << "Cannot find enough valid primes for scalingModSize\n";
        };

        logQP = computeLogPQ(scalingModSize, firstModSize, numPrimes);
        newRingDimension = lbcrypto::StdLatticeParm::FindRingDim(lbcrypto::HEStd_ternary, lbcrypto::HEStd_128_classic, logQP);
        
        if (newRingDimension == startDimension) {
          std::cerr << "Smaller possible. Set new RingDimension: " << newRingDimension << std::endl;
          startDimension = newRingDimension;
        } else {
          std::cerr << "Smaller not possible. Final RingDimension: " << 2 * startDimension << std::endl;
          ringDimension = 2 * startDimension;
          break;
        }
      }
    }

    // annotate mgmt::OpenfheParamsAttr to func::FuncOp containing the genericOp
    auto *funcOp = genericOp->getParentOp();
    auto openfheParamAttr = mgmt::OpenfheParamsAttr::get(
        funcOp->getContext(), multiplicativeDepth, scalingModSize, firstModSize);
    funcOp->setAttr(mgmt::MgmtDialect::kArgOpenfheParamsAttrName, openfheParamAttr);
  });
}

}  // namespace heir
}  // namespace mlir



// int findRingDimension() {

//   while (true) {
//     int startDimension = 16384;

//     scale, first = computeModuli()

//     logPQ = computeLogPQ 
//     dimension = computeDimension(logPQ)

//     if (dimension != startDimension) {
//       startDimension = dimension
//     }

//     if (dimension == startDimension) {
//       dimension /= 2
//       scale, first computeModuli()
//       logPQ = computeLogPQ
//       dimension = computeDimension(logPQ)
//       break;
//     }
//   }
// }