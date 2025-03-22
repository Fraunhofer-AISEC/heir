#include "OperationCountAnalysis.h"
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <utility>

#include "lib/Analysis/LevelAnalysis/LevelAnalysis.h"
#include "lib/Dialect/BGV/IR/BGVDialect.h"
#include "lib/Dialect/BGV/IR/BGVAttributes.h"
#include "lib/Dialect/BGV/IR/BGVEnums.h"
#include "lib/Dialect/Mgmt/IR/MgmtOps.h"
#include "lib/Dialect/Mgmt/IR/MgmtAttributes.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "src/core/include/math/hal/vector.h"                //from @openfhe
#include "src/core/include/math/nbtheory.h"                //from @openfhe
#include "src/core/include/lattice/stdlatticeparms.h"       //from @openfhe
#include "src/pke/include/scheme/scheme-utils.h"            //from @openfhe
#include "src/core/include/math/hal/nativeintbackend.h"        //from @openfhe
#include "llvm/include/llvm/ADT/TypeSwitch.h"              // from @llvm-project
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
    throw std::runtime_error("Could not find valid primes! FirstModSize or scalingModSize exeed maximum bit size!");
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
        throw std::runtime_error("Could not find valid primes for firstMod!");
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
      bool allFound = true;
      for (int i = 1; i < numPrimes; i++) {
        q = lbcrypto::PreviousPrime<lbcrypto::NativeInteger>(q, modulusOrder);
        if (q == firstMod) {
          firstModSize += 1;
          allFound = false;
          break;
        }
      }
      if (allFound || numPrimes == 1) {
        return {firstModSize, scalingModSize};
      }
    } catch (lbcrypto::OpenFHEException &e) {
      scalingModSize += 1;
    }
  }
  throw std::runtime_error("Could not find valid primes for scalingMod!");
}

static std::vector<double> computeBoundChain(
    const std::vector<double> &moduli,
    const std::vector<OperationCount> &levelOpCounts, NoiseBounds noiseBounds) {
  int ciphertextCount = levelOpCounts[moduli.size()].getCiphertextCount();
  int keySwitchCount = levelOpCounts[moduli.size()].getKeySwitchCount();

  std::vector<double> bound(moduli.size());
  bound[0] = ciphertextCount * (noiseBounds.boundClean + keySwitchCount * noiseBounds.addedNoiseKeySwitching);
  
  for (int i = 0; i < moduli.size() - 1; ++i) {
    int ciphertextCount = levelOpCounts[moduli.size() - 1 - i].getCiphertextCount();
    int keySwitchCount = levelOpCounts[moduli.size() - 1 - i].getKeySwitchCount();
    
    // No multiplication for B_clean
    double a = ciphertextCount * (bound[i] * bound[i] + keySwitchCount * noiseBounds.addedNoiseKeySwitching);
    bound[i + 1] = noiseBounds.boundScale + (a / moduli[i]);
  }

  return bound;
}

static std::vector<double> computeBoundChainFixed(
    double scalingMod, int numPrimes, const std::vector<OperationCount> &levelOpCounts,
    NoiseBounds noiseBounds) {

  std::vector<double> moduli(numPrimes - 1, scalingMod);
      
  return computeBoundChain(moduli, levelOpCounts, noiseBounds);
}

static double computeObjectiveFunction(
    double scalingMod, int ringDimension, int plaintextModulus,
    const std::vector<OperationCount> &levelOpCounts, int numPrimes,
    NoiseBounds noiseBounds) {

  std::vector<double> bound =
      computeBoundChainFixed(scalingMod, numPrimes, levelOpCounts, noiseBounds);

  double firstMod = 2 * bound.back();

  return scalingMod + firstMod;
}

static double computeFirstModSizeFromChain(
    double p, int ringDimension, int plaintextModulus,
    const std::vector<OperationCount> &levelOpCounts, int numPrimes,
    NoiseBounds noiseBounds) {
  std::vector<double> bound =
      computeBoundChainFixed(p, numPrimes, levelOpCounts,noiseBounds);
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

// Calculate objective function: max(p_list) + q where q = 2 * B[N]
static std::tuple<double, double, std::vector<double>> computeObjective(
    const std::vector<double>& moduli,
    const std::vector<OperationCount>& levelOpCounts,
    NoiseBounds noiseBounds) {
  auto bounds = computeBoundChain(moduli, levelOpCounts, noiseBounds);
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
  
  auto boundsOld = computeBoundChain(moduli, levelOpCounts, noiseBounds);
  double target = boundsOld[currentIndex + offset + 1];
  
  if (target - noiseBounds.boundScale <= 0) {
    return {};
  }
  
  auto boundsTemp = computeBoundChain(newModuli, levelOpCounts, noiseBounds);
  
  double x = boundsTemp[currentIndex + offset];
  int ciphertextCount = levelOpCounts[numberModuli - (currentIndex + offset) - 1].getCiphertextCount();
  int keySwitchCount = levelOpCounts[numberModuli - (currentIndex + offset) - 1].getKeySwitchCount();
  
  double newPartnerModuliValue = ciphertextCount * (x * x + keySwitchCount * noiseBounds.addedNoiseKeySwitching) / (target - noiseBounds.boundScale);
  
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
  
  auto boundsOld = computeBoundChain(moduli, levelOpCounts, noiseBounds);

  auto bound = boundsOld[currentIndex + 1];

  for (int i = currentIndex; i > currentIndex - offset; i--) {
    int ciphertextCount = levelOpCounts[numberModuli - i - 1].getCiphertextCount();
    int keySwitchCount = levelOpCounts[numberModuli - i - 1].getKeySwitchCount();
   
    double numerator = newModuli[i] * (bound - noiseBounds.boundScale);
    double denominator = ciphertextCount;
    double insideSqrt = (numerator / denominator) - (keySwitchCount * noiseBounds.addedNoiseKeySwitching);

    bound = sqrt(insideSqrt);
  }
  
  double x = boundsOld[currentIndex - offset];
  int ciphertextCount = levelOpCounts[numberModuli - (currentIndex - offset) - 1].getCiphertextCount();
  int keySwitchCount = levelOpCounts[numberModuli - (currentIndex - offset) - 1].getKeySwitchCount();
  
  double newPartnerModuliValue = ciphertextCount * (x * x + keySwitchCount * noiseBounds.addedNoiseKeySwitching) / (bound - noiseBounds.boundScale);
  
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
  auto boundsOld = computeBoundChain(moduli, levelOpCounts, noiseBounds);
  
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

  double x = boundsOld[numberModuli - offset - 1];
  int ciphertextCount = levelOpCounts[offset].getCiphertextCount();
  int keySwitchCount = levelOpCounts[offset].getKeySwitchCount();
  
  double newPartnerModuliValue = ciphertextCount * (x * x + keySwitchCount * noiseBounds.addedNoiseKeySwitching) / (bound - noiseBounds.boundScale);
  
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
        computeBoundChainFixed(scalingMod, numPrimes, levelOpCounts, noiseBounds);
    return std::all_of(bounds.begin(), bounds.end(), 
                      [](double b) { return !std::isinf(b) && !std::isnan(b); });
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

static int computeLogPQ(const std::vector<int> &moduli) {
  auto numPartQ = ComputeNumLargeDigits(0, moduli.size() - 1);
  auto logQ = std::accumulate(moduli.begin(), moduli.end(), 0);
  auto logP = ceil(ceil(static_cast<double>(logQ) / numPartQ) / kMaxBitSize) *
              kMaxBitSize;
  return logP + logQ;
};

static int computeRingDimension(const std::vector<int> &moduli) {
  auto logQP = computeLogPQ(moduli);
  return lbcrypto::StdLatticeParm::FindRingDim(
      lbcrypto::HEStd_ternary, lbcrypto::HEStd_128_classic, logQP);
};

static NoiseBounds calculateBoundParams(int ringDimension, int plaintextModulus,
                                        int numPrimes) {
  auto phi = ringDimension;  // Pessimistic
  auto t = plaintextModulus;
  auto d = 6.0;

  auto vKey = 2.0 / 3.0;   
  auto vErr = 3.19 * 3.19; 

  auto boundScale = d * t * sqrt((phi / 12.0) * (1.0 + (phi * vKey)));

  auto boundClean = d * t * sqrt(phi * (1.0 / 12.0 + 2 * phi * vErr * vKey + vErr));

  auto boundKeySwitch = d * t * phi * sqrt(vErr / 12.0);

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

  moduli.reserve(rebalanced.size());
  for (const auto& p : rebalanced) {
    moduli.push_back(ceil(log2(p)));
  }
  return moduli;
}

using BigInteger = bigintbackend::BigInteger;

static std::vector<int64_t> computePiModuli(const std::vector<int64_t> &qi,
                                            int ringDimension,
                                            int plaintextModulus) {
  // Following OpenFHE's approach for extension moduli in HYBRID key switching
  std::vector<int64_t> pi;

  // Calculate auxiliary primes bit size (auxBits)
  // In OpenFHE, this is typically set to 60 bits for 128-bit security
  int auxBits = heir::kMaxBitSize;

  // Find number of digits/partitions of Q (similar to numPartQ in OpenFHE)
  auto numPartQ = ComputeNumLargeDigits(0, qi.size() - 1);

  // Group qi into partitions as done in HYBRID
  std::vector<BigInteger> moduliPartQ;
  moduliPartQ.resize(numPartQ);

  // Calculate partitions similar to OpenFHE (ceil(sizeQ/numPartQ) towers per
  // digit)
  uint32_t a = ceil(static_cast<double>(qi.size()) / numPartQ);

  // Compute the composite digits PartQ = Q_j
  for (uint32_t j = 0; j < numPartQ; j++) {
    moduliPartQ[j] = BigInteger(1);
    for (uint32_t i = a * j; i < (j + 1) * a; i++) {
      if (i < qi.size()) moduliPartQ[j] *= qi[i];
    }
  }

  // Find number and size of individual special primes using the max bit length
  uint32_t maxBits = 0;
  for (uint32_t j = 0; j < numPartQ; j++) {
    uint32_t bits = moduliPartQ[j].GetLengthForBase(2);
    if (bits > maxBits) maxBits = bits;
  }

  // Select number of primes in auxiliary CRT basis
  uint32_t sizeP = ceil(static_cast<double>(maxBits) / auxBits);

  // Start with first prime as done in OpenFHE
  lbcrypto::NativeInteger firstP =
      lbcrypto::FirstPrime<lbcrypto::NativeInteger>(auxBits, ringDimension);
  lbcrypto::NativeInteger pPrev = firstP;

  // Generate each auxiliary prime
  for (uint32_t i = 0; i < sizeP; i++) {
    // The following loop makes sure that moduli in P and Q are different
    lbcrypto::NativeInteger currentP;
    bool foundInQ;
    do {
      currentP =
          lbcrypto::PreviousPrime<lbcrypto::NativeInteger>(pPrev, ringDimension);
      foundInQ = false;
      for (long j : qi) {
        if (currentP.ConvertToInt() == j) {
          foundInQ = true;
          break;
        }
      }
      pPrev = currentP;
    } while (foundInQ);

    pi.push_back(currentP.ConvertToInt());
  }

  return pi;
}

static std::vector<int64_t> computeQiModuliFromSizes(
    const std::vector<int> &moduliSizes, int ringDimension,
    int plaintextModulus) {
  std::vector<int64_t> qi;
  qi.reserve(moduliSizes.size());

  uint64_t modulusOrder = computeModulusOrder(ringDimension, plaintextModulus);

  // Process each modulus size in sequence
  lbcrypto::NativeInteger currentPrime = 0;
  for (int moduliSize : moduliSizes) {
    // First modulus or reset needed, get the last prime of this size
    currentPrime = lbcrypto::FirstPrime<lbcrypto::NativeInteger>(moduliSize, modulusOrder);
    // Get the previous prime with appropriate size
    while (true) {
       // Check for collisions with previously selected primes
      bool foundDuplicate = false;
       for (const auto& existingPrime : qi) {
        if (existingPrime == currentPrime.ConvertToInt()) {
          currentPrime = lbcrypto::NextPrime<lbcrypto::NativeInteger>(currentPrime, modulusOrder);
          foundDuplicate = true;
          break;
        }
      }
      if (!foundDuplicate) {
        break;
      }
    }
    
    qi.push_back(currentPrime.ConvertToInt());
  }

  return qi;
}

static void annotateSchemeParam(Operation *op, const uint64_t plaintextModulus,
                         const uint64_t ringDimension, const std::vector<int>& moduliSizes) {
  // Compute qi moduli from the vector of moduli sizes
  std::vector<int64_t> qi =
      computeQiModuliFromSizes(moduliSizes, ringDimension, plaintextModulus);

  // Compute pi moduli (extension moduli)
  std::vector<int64_t> pi =
      computePiModuli(qi, ringDimension, plaintextModulus);

  // Set the scheme parameters attribute
  op->setAttr(bgv::BGVDialect::kSchemeParamAttrName,
              bgv::SchemeParamAttr::get(
                  op->getContext(), log2(ringDimension),
                  DenseI64ArrayAttr::get(op->getContext(), ArrayRef<int64_t>(qi)),
                  DenseI64ArrayAttr::get(op->getContext(), ArrayRef<int64_t>(pi)),
                  plaintextModulus, bgv::BGVEncryptionType::pk, bgv::BGVEncryptionTechnique::standard));
}

static void annotateOpenfheParams(secret::GenericOp genericOp,
                                  int multiplicativeDepth, int ringDimension,
                                  const std::vector<int> &moduliSizes) {
  auto *funcOp = ((Operation*) genericOp)->getParentOp();

  // Compute the first and scaling moduli sizes
  int firstModSize = moduliSizes[0];
  int scalingModSize = *std::max_element(moduliSizes.begin() + 1, moduliSizes.end());
    
  auto openfheParamAttr = mgmt::OpenfheParamsAttr::get(
    funcOp->getContext(), multiplicativeDepth, ringDimension,
    scalingModSize, firstModSize, 0, 0);

  funcOp->setAttr(mgmt::MgmtDialect::kArgOpenfheParamsAttrName,
          openfheParamAttr);
}

static std::vector<int> createModuliSizeChain(int firstModSize, int scalingModSize, int numPrimes) {
  std::vector<int> moduli;
  moduli.push_back(firstModSize);
  for (int i = 1; i < numPrimes; i++) {
    moduli.push_back(scalingModSize);
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

    auto computeModuliSizes([&](int ringDimension) -> std::vector<int> {
      if (numPrimes == 1) {
        auto noiseBounds = calculateBoundParams(ringDimension, plaintextModulus, numPrimes);
        int firstModSize = ceil(1 + log2(levelOpCounts[0].getCiphertextCount()) + log2(noiseBounds.boundClean + (levelOpCounts[0].getKeySwitchCount() * noiseBounds.addedNoiseKeySwitching)));
        return {firstModSize};
      } 
      try {
        int firstModSize = 0;
        int scalingModSize = 0;
        if (algorithm == "BISECTION") {
          computeModuliSizesBisection(firstModSize, scalingModSize, ringDimension,
                                      plaintextModulus, levelOpCounts, numPrimes);
          auto modSizes =
              findValidPrimes(scalingModSize, firstModSize, numPrimes,
                              ringDimension, plaintextModulus);
          return createModuliSizeChain(modSizes.first, modSizes.second,
                                       numPrimes);
        }
        if (algorithm == "CLOSED") {
          computeModuliSizesClosed(firstModSize, scalingModSize, ringDimension,
                                  plaintextModulus, levelOpCounts, numPrimes);
          auto modSizes =
              findValidPrimes(scalingModSize, firstModSize, numPrimes,
                              ringDimension, plaintextModulus);
          return createModuliSizeChain(modSizes.first, modSizes.second,
                                       numPrimes);
        }
        if (algorithm == "BALANCING") {
          //TODO: Add validation of moduli sizes
          return computeModuliSizesBalancing(ringDimension, plaintextModulus, 
                                                    levelOpCounts, numPrimes);
        }
      } catch (const std::runtime_error& e) {
        genericOp->emitOpError() << "Parameter optimization failed: " << e.what();
        return {};
      }
      return {};
    });

    if (!isRingDimensionSet) {
      ringDimension = 16384;
    }

    auto newRingDimension = ringDimension;

    std::vector<int> moduli;
    while (true) {
      // Compute param sizes for HYBRID Key Switching
      moduli = computeModuliSizes(ringDimension);

      if (isRingDimensionSet) {
        break;
      }
  
      newRingDimension = computeRingDimension(moduli);
  
      if (newRingDimension == ringDimension) {
        // Try smaller ring dimension
        int smallerDimension = ringDimension / 2;
  
        auto newModuli = computeModuliSizes(smallerDimension);

        if (moduli.size() == 0) {
          break;
        }
        newRingDimension = computeRingDimension(newModuli);
  
        if (newRingDimension == smallerDimension) {
          ringDimension = smallerDimension;
          moduli = newModuli;
        } else {
          // No further improvement possible
          break;
        }

      } else {
        // New ring dimension is smaller
        ringDimension = newRingDimension;
      }
   }

   annotateSchemeParam(top, plaintextModulus, ringDimension, moduli);
   annotateOpenfheParams(genericOp, multiplicativeDepth, ringDimension, moduli);
  });
}
}  // namespace heir
}  // namespace mlir