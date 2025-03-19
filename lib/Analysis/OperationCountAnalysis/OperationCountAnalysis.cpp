#include "lib/Analysis/OperationCountAnalysis/OperationCountAnalysis.h"
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include <any>
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

struct NoiseBounds {
  double boundScale;
  double boundClean;
  double addedNoiseKeySwitching;
};

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

static std::pair<uint64_t, uint64_t> findValidPrimes(int scalingModSize, int firstModSize,
                                        int numPrimes, int ringDimension,
                                        int plaintextModulus) {
  if (firstModSize > kMaxBitSize || scalingModSize > kMaxBitSize) {
    return {0, 0};
  }                           

  uint64_t modulusOrder = computeModulusOrder(ringDimension, plaintextModulus);

  lbcrypto::NativeInteger firstMod = 0;

  // Find valid firstModulusSize and respective firstModulus prime
  while (true) {
    try {
      firstMod = lbcrypto::LastPrime<lbcrypto::NativeInteger>(firstModSize, modulusOrder);
      break;
    } catch (lbcrypto::OpenFHEException &e) {
      firstModSize += 1;
      if (firstModSize > kMaxBitSize) {
        return {0, 0};
      }
    }
  }

  while (scalingModSize <= kMaxBitSize) {
    try {
      lbcrypto::NativeInteger q;
      if (firstModSize == scalingModSize){
        firstMod = lbcrypto::LastPrime<lbcrypto::NativeInteger>(firstModSize,
                                                            modulusOrder);
        q = firstMod;
      } else {
        q = lbcrypto::LastPrime<lbcrypto::NativeInteger>(scalingModSize,
                                                         modulusOrder);
      }
      bool allFound = false;
      for (int i = 1; i < numPrimes; i++) {
        q = lbcrypto::PreviousPrime<lbcrypto::NativeInteger>(q, modulusOrder);
        if (q == firstMod) {
          firstModSize += 1;
          break;
        }
        allFound = true;
        }
      if (allFound || numPrimes == 1) {
        return {firstModSize, scalingModSize};
      }
    } catch (lbcrypto::OpenFHEException &e) {
      scalingModSize += 1;
    }
  }
  return {0,0};
}

static std::vector<double> computeBoundChain(
    double scalingMod, int numPrimes, const std::vector<OperationCount> &levelOpCounts,
    NoiseBounds noiseBounds) {

  std::vector<double> bound(numPrimes);
  bound[0] = noiseBounds.boundClean;
  
  for (int i = 0; i < numPrimes - 1; ++i) {
    int ciphertextCount = levelOpCounts[numPrimes - i - 1].getCiphertextCount();
    int keySwitchCount = levelOpCounts[numPrimes - i - 1].getKeySwitchCount();

    double a = ciphertextCount * (bound[i] * bound[i] + keySwitchCount * noiseBounds.addedNoiseKeySwitching);
    bound[i + 1] = noiseBounds.boundScale + (a / scalingMod);
  }

  return bound;
}

static double computeObjectiveFunction(
    double scalingMod, int ringDimension, int plaintextModulus,
    const std::vector<OperationCount> &levelOpCounts, int numPrimes,
    NoiseBounds noiseBounds) {

  std::vector<double> bound =
      computeBoundChain(scalingMod, numPrimes, levelOpCounts, noiseBounds);

  double firstMod = 2 * bound[numPrimes - 1];

  return scalingMod + firstMod;
}

static double computeFirstModSizeFromChain(
    double p, int ringDimension, int plaintextModulus,
    const std::vector<OperationCount> &levelOpCounts, int numPrimes,
    NoiseBounds noiseBounds) {
  std::vector<double> bound =
      computeBoundChain(p, numPrimes, levelOpCounts,noiseBounds);
  auto firstModSize = ceil(log2(2 * bound[numPrimes - 1]));
  if (std::isnan(firstModSize) || std::isinf(firstModSize)){
    return 0;
  }
  return firstModSize;
}

static double derivativeObjective(
    double p, int ringDimension, int plaintextModulus,
    const std::vector<OperationCount> &levelOpCounts, int numPrimes,
    NoiseBounds noiseBounds, double relStep = 1e-6) {
  double h = relStep * p;
  auto highObjective = computeObjectiveFunction(p + h, ringDimension,
                                               plaintextModulus, levelOpCounts,
                                               numPrimes, noiseBounds);
  auto lowObjective = computeObjectiveFunction(p - h, ringDimension,
                                              plaintextModulus, levelOpCounts,
                                              numPrimes, noiseBounds); 
  return (highObjective - lowObjective) / (2 * h);
}

static std::vector<double> computeChain(
    const std::vector<double>& moduli, 
    const std::vector<OperationCount>& levelOpCounts,
    NoiseBounds noiseBounds) {

  int numberModuli = moduli.size() + 1; // Plus one due to the first modulus
  std::vector<double> bounds(numberModuli);
  bounds[0] = noiseBounds.boundClean;
  
  for (int i = 0; i < numberModuli - 1; ++i) {
    int ciphertextCount = levelOpCounts[numberModuli - i - 1].getCiphertextCount();
    int keySwitchCount = levelOpCounts[numberModuli - i - 1].getKeySwitchCount();
    
    double boundSquare;
    try {
      boundSquare = bounds[i] * bounds[i];
    } catch (...) {
      boundSquare = std::numeric_limits<double>::infinity();
    }
    
    double A = ciphertextCount * (boundSquare + keySwitchCount * noiseBounds.addedNoiseKeySwitching);
    bounds[i + 1] = noiseBounds.boundScale + (A / moduli[i]);
  }
  
  return bounds;
}

// Calculate objective function: max(p_list) + q where q = 2 * B[N]
static std::tuple<double, double, std::vector<double>> computeObjective(
    const std::vector<double>& moduli,
    const std::vector<OperationCount>& levelOpCounts,
    NoiseBounds noiseBounds) {
  auto bounds = computeChain(moduli, levelOpCounts, noiseBounds);
  double firstMod = 2 * bounds.back();
  
  double sumModuli = 0;
  for (auto mod : moduli) {
    sumModuli += mod;
  }
  
  return {sumModuli + firstMod, firstMod, bounds};
}

// Forward candidate update
static std::vector<double> candidateForward(
    const std::vector<double>& moduli,
    const std::vector<OperationCount>& levelOpCounts,
    int currentIndex, double factor, int offset,
    NoiseBounds noiseBounds) {
    
  int numberModuli = moduli.size();
  
  std::vector<double> newModuli = moduli;
  double newModuliValue = factor * moduli[currentIndex];
  
  if (newModuliValue <= 0) {
    return {};
  }
  
  newModuli[currentIndex] = newModuliValue;
  
  auto boundsOld = computeChain(moduli, levelOpCounts, noiseBounds);
  double target = boundsOld[currentIndex + offset + 1];
  
  if (target - noiseBounds.boundScale <= 0) {
    return {};
  }
  
  auto boundsTemp = computeChain(newModuli, levelOpCounts, noiseBounds);
  
  double X = boundsTemp[currentIndex + offset];
  int ciphertextCount = levelOpCounts[numberModuli - (currentIndex + offset) - 1].getCiphertextCount();
  int keySwitchCount = levelOpCounts[numberModuli - (currentIndex + offset) - 1].getKeySwitchCount();
  
  double newPartnerModuliValue = ciphertextCount * (X * X + keySwitchCount * noiseBounds.addedNoiseKeySwitching) / (target - noiseBounds.boundScale);
  
  if (newPartnerModuliValue <= 0) {
    return {};
  }
  
  newModuli[currentIndex + offset] = newPartnerModuliValue;
  return newModuli;
}

static std::vector<double> candidateBackward(
    const std::vector<double>& moduli,
    const std::vector<OperationCount>& levelOpCounts,
    int currentIndex, double factor, int offset,
    NoiseBounds noiseBounds) {
    
  int numberModuli = moduli.size();

  std::vector<double> newModuli = moduli;
  double newModuliValue = factor * moduli[currentIndex];
  
  if (newModuliValue <= 0) {
    return {};
  }
  
  newModuli[currentIndex] = newModuliValue;
  
  auto boundsOld = computeChain(moduli, levelOpCounts, noiseBounds);

  auto bound = boundsOld[currentIndex + 1];

  for (int i = currentIndex; i > currentIndex - offset; i--) {
    int ciphertextCount = levelOpCounts[numberModuli - i - 1].getCiphertextCount();
    int keySwitchCount = levelOpCounts[numberModuli - i - 1].getKeySwitchCount();
   
    double numerator = newModuli[i] * (bound - noiseBounds.boundScale);
    double denominator = ciphertextCount;
    double insideSqrt = (numerator / denominator) - (keySwitchCount * noiseBounds.addedNoiseKeySwitching);

    bound = sqrt(insideSqrt);
  }
  
  double X = boundsOld[currentIndex - offset];
  int ciphertextCount = levelOpCounts[numberModuli - (currentIndex - offset) - 1].getCiphertextCount();
  int keySwitchCount = levelOpCounts[numberModuli - (currentIndex - offset) - 1].getKeySwitchCount();
  
  double newPartnerModuliValue = ciphertextCount * (X * X + keySwitchCount * noiseBounds.addedNoiseKeySwitching) / (bound - noiseBounds.boundScale);
  
  if (newPartnerModuliValue <= 0) {
    return {};
  }
  
  newModuli[currentIndex - offset] = newPartnerModuliValue;
  return newModuli;
}

static std::vector<double> candidateFirstModUpdate(
    const std::vector<double>& moduli,
    const std::vector<OperationCount>& levelOpCounts,
    double factor, int offset,
    NoiseBounds noiseBounds) {
    
  int numberModuli = moduli.size() + 1; // Plus one due to the first modulus
  auto boundsOld = computeChain(moduli, levelOpCounts, noiseBounds);
  
  double currentFirstMod = 2 * boundsOld.back();
  double newFirstMod = factor * currentFirstMod;

  auto bound = newFirstMod / 2;

  for (int i = numberModuli - 2; i >= numberModuli - offset; i--) {
    int ciphertextCount = levelOpCounts[numberModuli - i - 1].getCiphertextCount();
    int keySwitchCount = levelOpCounts[numberModuli - i - 1].getKeySwitchCount();
   
    double numerator = moduli[i] * (bound - noiseBounds.boundScale);
    double denominator = ciphertextCount;
    double insideSqrt = (numerator / denominator) - (keySwitchCount * noiseBounds.addedNoiseKeySwitching);

    bound = sqrt(insideSqrt);
  }

  double X = boundsOld[numberModuli - offset - 1];
  int ciphertextCount = levelOpCounts[offset].getCiphertextCount();
  int keySwitchCount = levelOpCounts[offset].getKeySwitchCount();
  
  double newPartnerModuliValue = ciphertextCount * (X * X + keySwitchCount * noiseBounds.addedNoiseKeySwitching) / (bound - noiseBounds.boundScale);
  
  if (newPartnerModuliValue <= 0) {
    return {};
  }
  
  std::vector<double> newModuli = moduli;
  newModuli[numberModuli - offset - 1] = newPartnerModuliValue;
  
  return newModuli;
}

static bool anyModuliNegative(const std::vector<double>& moduli) {
  return std::any_of(moduli.begin(), moduli.end(), [](double val) { return val <= 0; });
}


static std::vector<double> rebalanceSingleModulus(
    const std::vector<double>& moduli,
    const std::vector<OperationCount>& levelOpCounts,
    int currentModuliIndex, double factor,
   NoiseBounds noiseBounds) {
    
  int numberModuli = moduli.size() + 1; // Plus one due to the first modulus
  std::vector<std::tuple<std::vector<double>, double>> candidates;
  
  // Forward updates
  for (int d = 1; d < numberModuli - 1 - currentModuliIndex; ++d) {
    auto cand = candidateForward(moduli, levelOpCounts, currentModuliIndex, factor, d, 
                                noiseBounds);
    if (!cand.empty()) {
      if (anyModuliNegative(cand)) {
        continue;
      }
      auto [objective, firstMod, __] = computeObjective(cand, levelOpCounts, 
                                              noiseBounds);
      if (log2(firstMod) <= 0) {
        continue;
      }
      candidates.emplace_back(cand, objective);
    }
  }

  // Backward updates
    for (int d = 1; d <= currentModuliIndex; ++d) {
      auto cand = candidateBackward(moduli, levelOpCounts, currentModuliIndex, factor, d, noiseBounds);
      if (!cand.empty()) {
        if (anyModuliNegative(cand)) {
          continue;
        }
        auto [objective, firstMod, __] = computeObjective(cand, levelOpCounts, 
                                                noiseBounds);
        if (log2(firstMod) <= 0) {
          continue;
        }
        candidates.emplace_back(cand, objective);
      }
    }

  if (candidates.empty()) {
    return {};
  }
  
  // Find the candidate with lowest objective value
  auto minCandidate = std::min_element(candidates.begin(), candidates.end(),
                                [](const auto& a, const auto& b) {
                                  return std::get<1>(a) < std::get<1>(b);
                                });
  
  return std::get<0>(*minCandidate);
}

static std::vector<double> rebalancingModuli(
    const double pInit,
    const std::vector<OperationCount>& levelOpCounts,
    NoiseBounds noiseBounds,
    int maxIter = 100, double tolerance = 0.01, double eps = 1e-6) {
  
  int numberModuli = levelOpCounts.size();
  std::vector<double> moduliCurrent(numberModuli - 1,pInit);
  
  std::vector<double> candidateFactors = {0.5, 0.75, 0.9, 1.1, 1.25, 1.5, 2.0};

  auto [objectiveCurrent, firstModCurrent, _] =
      computeObjective(moduliCurrent, levelOpCounts, noiseBounds);

  int globalIter = 0;
  while (globalIter < maxIter) {
    double bestObjective = objectiveCurrent;
    double bestFirstMod = firstModCurrent;
    std::vector<double> bestCandidate;
    
    // scaling moduli-updates
    for (int i = 0; i < numberModuli - 1; ++i) {
      for (double factor : candidateFactors) {
        auto candidate = rebalanceSingleModulus(moduliCurrent, levelOpCounts, i, factor, 
                                     noiseBounds);
        if (candidate.empty()) {
          continue;
        }

        auto [objective, firstMod, __] =
            computeObjective(candidate, levelOpCounts, noiseBounds);

        if (objective < bestObjective) {
          bestObjective = objective;
          bestCandidate = candidate;
          bestFirstMod = firstMod;
        }
      }
    }
    
    // first mod-updates
    for (double factor : candidateFactors) {
      for (int offset = 1; offset < numberModuli - 1; ++offset) {
        auto candidate =
            candidateFirstModUpdate(moduliCurrent, levelOpCounts, factor, offset, noiseBounds);
        if (candidate.empty() || anyModuliNegative(candidate)) {
          continue;
        }

        auto [objective, firstMod, __] =
            computeObjective(candidate, levelOpCounts, noiseBounds);
        
        if (log2(firstMod) <= 0) {
          continue;
        }

        if (objective < bestObjective) {
          bestObjective = objective;
          bestCandidate = candidate;
          bestFirstMod = firstMod;
        }
      }
    }
    
    double improvement = objectiveCurrent - bestObjective;
    if (improvement > tolerance && !bestCandidate.empty()) {
      moduliCurrent = bestCandidate;
      objectiveCurrent = bestObjective;
      firstModCurrent = bestFirstMod;
    } else {
      break;
    }
    
    globalIter++;
  }
  
  std::vector<double> result;
  result.reserve(moduliCurrent.size() + 1);
  result.push_back(firstModCurrent);
  result.insert(result.end(), moduliCurrent.rbegin(), moduliCurrent.rend());
  
  return result;
}

static double findOptimalScalingModSizeBisection(
    int ringDimension, int plaintextModulus,
    const std::vector<OperationCount> &levelOpCounts, int numPrimes,
    NoiseBounds noiseBounds, double pLow = 2, double pHigh = pow(2.0, 60)) {
  auto checkBounds = [&](double scalingMod) {
    std::vector<double> bounds =
        computeBoundChain(scalingMod, numPrimes, levelOpCounts,noiseBounds);
    for (const auto& b : bounds) {
      if (std::isinf(b) || std::isnan(b)) {
        return false;
      }
    }
    return true;
  };

  // Increase pLow until all bounds in the chain are valid
  while (!checkBounds(pLow) && pLow < pHigh) {
    pLow *= 2.0;
    if (pLow > pHigh) {
      throw std::runtime_error("No valid lower bound found for bisection start");
    }
  }
  
  // Bisection
  while (log2(pHigh) - log2(pLow) > 1) {
    double pMid = (pLow + pHigh) / 2.0;
    if (derivativeObjective(pMid, ringDimension, plaintextModulus,
                            levelOpCounts, numPrimes, noiseBounds) < 0) {
      // lower then the minimizer.
      pLow = pMid;
    } else {
      // higher then the minimizer.
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

static NoiseBounds calculateBoundParams(int ringDimension, int plaintextModulus,
                                        int numPrimes) {
  auto phi = ringDimension;  // Pessimistic
  auto t = plaintextModulus;
  auto D = 6.0;

  auto vKey = 2.0 / 3.0;   
  auto vErr = 3.19 * 3.19; 

  auto boundScale = D * t * sqrt((phi / 12.0) * (1.0 + (phi * vKey)));

  auto boundClean = D * t * sqrt(phi * (1.0 / 12.0 + 2 * phi * vErr * vKey + vErr));

  auto boundKeySwitch = D * t * phi * sqrt(vErr / 12.0);

  auto f0 = 1;

  auto addedNoiseKeySwitching = f0 * boundKeySwitch + boundScale;

  return {boundScale, boundClean, addedNoiseKeySwitching};
}

static int getMaxLevel(secret::GenericOp *op) {
  int maxLevel = 0;

  op->getBody()->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (op->getNumResults() == 0) {
      return;
    }
    int level = getLevelFromMgmtAttr(op->getResult(0));
    maxLevel = std::max(maxLevel, level);
  });

  return maxLevel;
}

static std::vector<OperationCount> getLevelOpCounts(secret::GenericOp *op,
                                                    DataFlowSolver *solver,
                                                    int maxLevel) {
  std::vector<OperationCount> levelOpCounts;

  levelOpCounts.resize(maxLevel + 1, OperationCount(0, 0));

  // Second pass to populate the vector
  op->getBody()->walk<WalkOrder::PreOrder>([&](Operation *op) {
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

  return levelOpCounts;
}

static void computeModuliSizesBisection(int &firstModSize, int &scalingModSize,
                                        int ringDimension, int plaintextModulus,
                                        const std::vector<OperationCount> &levelOpCounts,
                                        int numPrimes) {
  auto noiseBounds = calculateBoundParams(ringDimension, plaintextModulus, numPrimes);

  try {
    double scalingMod = findOptimalScalingModSizeBisection(
        ringDimension, plaintextModulus, levelOpCounts, numPrimes, noiseBounds);
    
    firstModSize = computeFirstModSizeFromChain(
        scalingMod, ringDimension, plaintextModulus, levelOpCounts, numPrimes, noiseBounds);
    scalingModSize = ceil(log2(scalingMod));
  } catch (const std::runtime_error& e) {
    throw; // Re-throw the exception to be caught in annotateCountParams
  }
}

static void computeModuliSizesClosed(
    int &firstModSize, int &scalingModSize, int ringDimension,
    int plaintextModulus, const std::vector<OperationCount> &levelOpCounts,
    int numPrimes) {
 auto noiseBounds =  calculateBoundParams(ringDimension, plaintextModulus, numPrimes);

 // Compute OperationCounts over all levels
 OperationCount maxCounts(0, 0);
 for (auto count : levelOpCounts) {
   maxCounts = OperationCount::max(maxCounts, count);
 }

 auto boundOptimal =
     log2(noiseBounds.boundScale +
          sqrt(noiseBounds.boundScale * noiseBounds.boundScale + (maxCounts.getKeySwitchCount() *
                                          noiseBounds.addedNoiseKeySwitching)));

  if (boundOptimal < noiseBounds.boundClean) {
    boundOptimal =
    log2(noiseBounds.boundScale +
         sqrt(noiseBounds.boundClean * noiseBounds.boundClean + (maxCounts.getKeySwitchCount() *
                                         noiseBounds.addedNoiseKeySwitching)));
  }

 scalingModSize =
     ceil(1 + log2(maxCounts.getCiphertextCount()) + boundOptimal);
 firstModSize = ceil(1 + boundOptimal);
}

static std::vector<int> computeModuliSizesBalancing(
    int ringDimension, int plaintextModulus,
    const std::vector<OperationCount> &levelOpCounts, int numPrimes) {
  auto noiseBounds = calculateBoundParams(ringDimension, plaintextModulus, numPrimes);

  // Use bisection result as init value
  double pInit = findOptimalScalingModSizeBisection(
    ringDimension, plaintextModulus, levelOpCounts, numPrimes, noiseBounds);


  auto rebalanced = rebalancingModuli(pInit, levelOpCounts, noiseBounds);
  
  std::vector<int> moduli;

  for (const auto& p : rebalanced) {
    moduli.push_back(ceil(log2(p)));
  }
  return moduli;
}

void annotateCountParams(Operation *top, DataFlowSolver *solver,
                         int ringDimension, int plaintextModulus,
                         std::string algorithm) {
  top->walk<WalkOrder::PreOrder>([&](secret::GenericOp genericOp) {
    bool isRingDimensionSet = ringDimension != 0;

    auto maxLevel = getMaxLevel(&genericOp);
    auto levelOpCounts = getLevelOpCounts(&genericOp, solver, maxLevel);

    auto multiplicativeDepth = maxLevel;
    auto numPrimes = multiplicativeDepth + 1;

    auto computeModuliSizes([&](int ringDimension, int &firstModSize,
                               int &scalingModSize) {
      try {
        if (algorithm == "BISECTION") {
          computeModuliSizesBisection(firstModSize, scalingModSize, ringDimension,
                                      plaintextModulus, levelOpCounts, numPrimes);
          auto moduli = computeModuliSizesBalancing(
              ringDimension, plaintextModulus, levelOpCounts, numPrimes);
          // Print moduli information to cerr
          std::cerr << "Algorithm: BISECTION\n";
          std::cerr << "First modulus size: " << firstModSize << "\n";
          std::cerr << "Scaling modulus size: " << scalingModSize << "\n";
          std::cerr << "Balanced moduli: [";
          for (size_t i = 0; i < moduli.size(); ++i) {
            std::cerr << moduli[i];
            if (i < moduli.size() - 1) {
              std::cerr << ", ";
            }
          }
          std::cerr << "]\n";
        } else if (algorithm == "CLOSED") {
          computeModuliSizesClosed(firstModSize, scalingModSize, ringDimension,
                                  plaintextModulus, levelOpCounts, numPrimes);
          auto moduli = computeModuliSizesBalancing(
              ringDimension, plaintextModulus, levelOpCounts, numPrimes);
        } else if (algorithm == "BALANCING") {
          auto moduli = computeModuliSizesBalancing(ringDimension, plaintextModulus, 
                                                    levelOpCounts, numPrimes);
          // TODO: Temp fix to adapt to the existing code
          firstModSize = moduli[0];
          for (int i = 1; i < moduli.size(); i++) {
            if (moduli[i] > scalingModSize) {
              scalingModSize = moduli[i];
            }
          }
        }
      } catch (const std::runtime_error& e) {
        genericOp->emitOpError() << "Parameter optimization failed: " << e.what();
        return;
      }
  
      auto modSizes = findValidPrimes(scalingModSize, firstModSize, numPrimes, ringDimension, plaintextModulus);
      if (!modSizes.first || !modSizes.second) {
        genericOp->emitOpError() << "Could not find valid primes for modulus sizes!\n";
        return;
      }
      firstModSize = modSizes.first;
      scalingModSize = modSizes.second;
    });

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
        if (!smallerScalingModSize || !smallerFirstModSize) {
          break;
        }

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
      }
    }

    // annotate mgmt::OpenfheParamsAttr to func::FuncOp containing the genericOp
    auto *funcOp = ((Operation*) genericOp)->getParentOp();
    auto openfheParamAttr = mgmt::OpenfheParamsAttr::get(
        funcOp->getContext(), multiplicativeDepth, ringDimension,
        scalingModSize, firstModSize, 0, 0);
    funcOp->setAttr(mgmt::MgmtDialect::kArgOpenfheParamsAttrName,
                    openfheParamAttr);
  });
}

}  // namespace heir
}  // namespace mlir