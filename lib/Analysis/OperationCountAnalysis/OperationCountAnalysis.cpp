#include "lib/Analysis/OperationCountAnalysis/OperationCountAnalysis.h"
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <utility>

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

#include <cmath> // Required for math functions
#include <vector>

#define DEBUG_TYPE "operation-count-analysis"

namespace mlir {
namespace heir {

constexpr int kMaxBitSize = 60;

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

  while (minModSize < kMaxBitSize) {
    try {
      lbcrypto::LastPrime<lbcrypto::NativeInteger>(minModSize, modulusOrder);
      return minModSize;
    } catch (lbcrypto::OpenFHEException &e) {
      minModSize += 1;
    }
  }
  return 0;
}

static uint64_t findValidScalingModSize(int minModSize, int firstModSize,
                                        int numPrimes, int ringDimension,
                                        int plaintextModulus) {
  uint64_t modulusOrder = computeModulusOrder(ringDimension, plaintextModulus);

  lbcrypto::NativeInteger firstModulus = 0;
  if (firstModSize < minModSize) {
    firstModulus = lbcrypto::LastPrime<lbcrypto::NativeInteger>(firstModSize,
                                                                modulusOrder);
  }

  while (minModSize < kMaxBitSize) {
    try {
      auto q = lbcrypto::LastPrime<lbcrypto::NativeInteger>(minModSize,
                                                            modulusOrder);
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

// Computes based on the OpenFHE Model the chain of base bounds between level
static std::vector<double> computeBoundChain(
    double p, int numPrimes, const std::vector<OperationCount> &levelOpCounts,
    double boundClean, double boundScale, double addedNoiseKeySwitching) {

  std::vector<double> bound(numPrimes + 1);
  bound[0] = boundClean;
  
  for (int i = 0; i < numPrimes; ++i) {
    int ciphertextCount = levelOpCounts[numPrimes - i - 1].getCiphertextCount();
    int keySwitchCount = levelOpCounts[numPrimes - i - 1].getKeySwitchCount();

    double a = ciphertextCount * (bound[i] * bound[i] + keySwitchCount * addedNoiseKeySwitching);
    bound[i + 1] = boundScale + (a / p);
  }

  return bound;
}

static double computeObjectiveFunction(
    double p, int ringDimension, int plaintextModulus,
    const std::vector<OperationCount> &levelOpCounts, int numPrimes,
    double bClean, double bScale, double nuKS) {

  std::vector<double> bound =
      computeBoundChain(p, numPrimes, levelOpCounts, bClean, bScale, nuKS);

  double q = 2 * bound[numPrimes];

  return p + q;
}

static double computeFirstModSizeFromChain(
    double p, int ringDimension, int plaintextModulus,
    const std::vector<OperationCount> &levelOpCounts, int numPrimes,
    double bClean, double bScale, double nuKS) {
  std::vector<double> bound =
      computeBoundChain(p, numPrimes, levelOpCounts, bClean, bScale, nuKS);
  return ceil(log2(2 * bound[numPrimes]));
}

// Function to estimate the derivative of the objective function
static double derivativeObjective(
    double p, int ringDimension, int plaintextModulus,
    const std::vector<OperationCount> &levelOpCounts, int numPrimes,
    double bClean, double bScale, double nuKS, double rel_step = 1) {
  double h = rel_step * p;
  return (computeObjectiveFunction(p + h, ringDimension, plaintextModulus,
                                   levelOpCounts, numPrimes, bClean, bScale,
                                   nuKS) -
          computeObjectiveFunction(p - h, ringDimension, plaintextModulus,
                                   levelOpCounts, numPrimes, bClean, bScale,
                                   nuKS)) /
         (2 * h);
}

static double findOptimalScalingModSize(
    int ringDimension, int plaintextModulus,
    const std::vector<OperationCount> &levelOpCounts, int numPrimes,
    double bClean, double bScale, double nuKS, double tol = 1e-6,
    double pLow = 2, double pHigh = pow(2.0, 60)) {
  // Bisection on the derivative.
  while (log2(pHigh - pLow) > 1) {
    double pMid = (pLow + pHigh) / 2.0;
    if (derivativeObjective(pMid, ringDimension, plaintextModulus,
                            levelOpCounts, numPrimes, bClean, bScale,
                            nuKS) < 0) {
      // We are left of the minimizer.
      pLow = pMid;
    } else {
      // We are right of the minimizer.
      pHigh = pMid;
    }
  }
  return (pLow + pHigh) / 2.0;
}

static double computeLogPQ(int scalingModSize, int firstModSize,
                           int multDepth) {
  auto numPartQ = ComputeNumLargeDigits(0, multDepth);
  auto logQ = firstModSize + multDepth * scalingModSize;
  auto logP = ceil(ceil(static_cast<double>(logQ) / numPartQ) / kMaxBitSize) *
              kMaxBitSize;

  return logP + logQ;
};

static void calculateBoundParams(int ringDimension, int plaintextModulus,
                                 int numPrimes, double &bClean, double &bScale,
                                 double &nuKS) {
  auto phi = ringDimension;  // Pessimistic
  auto beta = pow(2.0, 10);  // TODO Replace by value set through developer
  auto t = plaintextModulus;
  auto D = 6.0;

  auto vKey = 2.0 / 3.0;    // TODO Make adjustable Currently for ternary
  auto vErr = 3.19 * 3.19;  // TODO Make adjustable

  bScale = D * t * sqrt((phi / 12.0) * (1.0 + (phi * vKey)));

  bClean = D * t * sqrt(phi * (1.0 / 12.0 + 2 * phi * vErr * vKey + vErr));

  auto boundKeySwitch = D * t * phi * sqrt(vErr / 12.0);

  auto f0 =
      beta * sqrt(numPrimes * log2(t * phi)) /
      (100.0 * beta * sqrt(log(numPrimes * pow(2.0, kMaxBitSize)) / log(beta)));

  nuKS = f0 * boundKeySwitch + bScale;
}

void annotateCountParams(Operation *top, DataFlowSolver *solver,
                         int ringDimension, int plaintextModulus,
                         std::string algorithm) {
  top->walk<WalkOrder::PreOrder>([&](secret::GenericOp genericOp) {
    bool isRingDimensionSet = ringDimension != 0;

    // Vector to store max operation counts per level
    std::vector<OperationCount> levelOpCounts;

    // First pass to determine max level
    int maxLevel = 0;
    genericOp.getBody()->walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (op->getNumResults() == 0) {
        return;
      }
      // Find the level for this operation's result
      int level = getLevelFromMgmtAttr(op->getResult(0));
      maxLevel = std::max(maxLevel, level);
    });

    // Initialize vector with appropriate size
    levelOpCounts.resize(maxLevel + 1, OperationCount(0, 0));

    // Second pass to populate the vector
    genericOp.getBody()->walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (op->getNumResults() == 0) {
        return;
      }
      auto *lattice =
          solver->lookupState<OperationCountLattice>(op->getResult(0));
      if (!lattice) {
        return;
      }

      auto count = lattice->getValue();
      if (!count.isInitialized()) {
        return;
      }

      // Get the level for the operation's result
      int level = getLevelFromMgmtAttr(op->getResult(0));

      // Update the max operation count for the level
      levelOpCounts[level] = OperationCount::max(levelOpCounts[level], count);
    });

    auto log2 = [](double x) { return log(x) / log(2); };

    auto multiplicativeDepth = maxLevel;
    auto numPrimes = multiplicativeDepth + 1;

    // Modified computeModuliSizes to use find_optimal_p and return optimal
    // firstModSize
    auto computeModuliSizes = [&](int ringDimension, int &firstModSize,
                                  int &scalingModSize) {
      double boundClean, boundScale, addedNoiseKeySwitching;
      calculateBoundParams(ringDimension, plaintextModulus, numPrimes,
                           boundClean, boundScale, addedNoiseKeySwitching);

      auto calculateModulusSizesBisection = [&](int &firstModSize,
                                                int &scalingModSize) {
        double p = findOptimalScalingModSize(
            ringDimension, plaintextModulus, levelOpCounts, numPrimes,
            boundClean, boundScale, addedNoiseKeySwitching);
        firstModSize = computeFirstModSizeFromChain(
            p, ringDimension, plaintextModulus, levelOpCounts, numPrimes,
            boundClean, boundScale, addedNoiseKeySwitching);
        scalingModSize = ceil(log2(p));
      };

      auto calculateModulusSizesClosedForm = [&](int &firstModSize,
                                                 int &scalingModSize) {
        // Compute OperationCounts over all levels
        OperationCount maxCounts(0, 0);
        for (auto count : levelOpCounts) {
          maxCounts = OperationCount::max(maxCounts, count);
        }
        auto boundOptimal =
            log2(boundScale +
                 sqrt(boundScale * boundScale + (maxCounts.getKeySwitchCount() *
                                                 addedNoiseKeySwitching)));

        scalingModSize =
            ceil(1 + log2(maxCounts.getCiphertextCount()) + boundOptimal);
        firstModSize = ceil(1 + boundOptimal);
      };

      if (algorithm == "BISECTION") {
        calculateModulusSizesBisection(firstModSize, scalingModSize);
      } else {
        calculateModulusSizesClosedForm(firstModSize, scalingModSize);
      }

      firstModSize =
          findValidFirstModSize(firstModSize, ringDimension, plaintextModulus);
      if (!firstModSize) {
        top->emitOpError() << "Could not find valid firstModSize\n";
        return;
      }

      scalingModSize =
          findValidScalingModSize(scalingModSize, firstModSize, numPrimes,
                                  ringDimension, plaintextModulus);
      if (!scalingModSize) {
        top->emitOpError() << "Could not find valid scalingModSize\n";
        return;
      }
    };

    auto computeRingDimension = [&](int scalingModSize, int firstModSize) {
      auto logQP = computeLogPQ(scalingModSize, firstModSize, numPrimes);
      return lbcrypto::StdLatticeParm::FindRingDim(
          lbcrypto::HEStd_ternary, lbcrypto::HEStd_128_classic, logQP);
    };

    if (!isRingDimensionSet) {
      ringDimension = 16384;
    }

    int firstModSize = 0;
    int scalingModSize = 0;

    auto newRingDimension = ringDimension;

    while (true) {
      // Compute param sizes for HYBRID Key Switching
      computeModuliSizes(ringDimension, firstModSize, scalingModSize);

      if (isRingDimensionSet) {
        break;
      }

      newRingDimension = computeRingDimension(scalingModSize, firstModSize);

      if (newRingDimension == ringDimension) {
        // Try smaller ring dimension
        int smallerDimension = ringDimension / 2;
        int smallerFirstModSize = 0;
        int smallerScalingModSize = 0;

        computeModuliSizes(smallerDimension, smallerFirstModSize,
                           smallerScalingModSize);
        newRingDimension =
            computeRingDimension(smallerScalingModSize, smallerFirstModSize);

        if (newRingDimension == smallerDimension) {
          ringDimension = smallerDimension;
        } else {
          // No further improvement possible
          break;
        }

      } else {
        // New ring dimension is smaller
        ringDimension = newRingDimension;
        break;
      }
    }

    // annotate mgmt::OpenfheParamsAttr to func::FuncOp containing the genericOp
    auto *funcOp = genericOp->getParentOp();
    auto openfheParamAttr = mgmt::OpenfheParamsAttr::get(
        funcOp->getContext(), multiplicativeDepth, ringDimension,
        scalingModSize, firstModSize, 0, 0);
    funcOp->setAttr(mgmt::MgmtDialect::kArgOpenfheParamsAttrName,
                    openfheParamAttr);
  });
}

}  // namespace heir
}  // namespace mlir