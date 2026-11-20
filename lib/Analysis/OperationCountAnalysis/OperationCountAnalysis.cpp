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
#include "src/pke/include/schemerns/rns-cryptoparameters.h" //from @openfhe
#include "src/core/include/math/hal/nativeintbackend.h"        //from @openfhe
#include "llvm/include/llvm/ADT/TypeSwitch.h"              // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"      // from @llvm-project
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"    // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"                 // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"

#include <cmath> // Required for math functions
#include <vector>

#define DEBUG_TYPE "operation-count-analysis"

namespace mlir {
namespace heir {

constexpr int kMaxBitSize = 60;

void OperationCountAnalysis::setToEntryState(OperationCountLattice *lattice) {
  Value value = lattice->getAnchor();

  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    if (auto genericOp = dyn_cast<secret::GenericOp>(blockArg.getOwner()->getParentOp())) {
      if (blockArg.getArgNumber() < genericOp.getNumOperands()) {
        Value genericOperand = genericOp.getOperand(blockArg.getArgNumber());
        auto *operandLattice = getLatticeElement(genericOperand);
        auto operandCount = operandLattice->getValue();
        if (operandCount.isInitialized()) {
          propagateIfChanged(lattice, lattice->join(operandCount));
          return;
        }
      }
      propagateIfChanged(lattice, lattice->join(OperationCount(1, 0, true)));
      return;
    }
  }

  if (isa<secret::SecretType>(value.getType())) {
    propagateIfChanged(lattice, lattice->join(OperationCount(1, 0, true)));
    return;
  }

  propagateIfChanged(lattice, lattice->join(OperationCount()));
}

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
      .Case<arith::AddIOp, arith::SubIOp>([&](auto addOp) {
        SmallVector<OpResult> secretResults;
        getSecretResults(op, secretResults);
        if (secretResults.empty()) {
          return;
        }

        OperationCount sumCount(0, 0, true);
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

        SmallVector<OpOperand *> secretOperands;
        getSecretOperands(op, secretOperands);

        if (secretOperands.size() == 2) {
          // ct * ct: noise is squared → reset to one fresh ciphertext
          propagate(mulOp->getResult(0), OperationCount(1, 0, 0, false));
        } else {
          // const * ct: noise scales linearly → preserve counts,
          //             increment constantMulCount
          auto count =
              operands[secretOperands[0]->getOperandNumber()]->getValue();
          if (!count.isInitialized()) {
            return;
          }
          propagate(mulOp->getResult(0),
                    OperationCount(count.getCiphertextCount(),
                                   count.getKeySwitchCount(),
                                   count.getConstantMulCount() + 1,
                                   count.isHighestLevel()));
        }
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
        propagate(modReduceOp.getResult(), OperationCount(1, 0));
      });

      return success();

  return mlir::success();
}

struct NoiseBounds {
  double boundScale;
  double boundClean;
  double addedNoiseKeySwitching;
  double boundConstant;

  double boundScaleSquared() {
    return boundScale * boundScale;
  }
};

template<typename T>
static int getBitSize(T value) {
  return static_cast<int>(std::floor(std::log2(value)) + 1);
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

static std::vector<double> computeBoundChain(
    const std::vector<double> &moduli,
    const std::vector<OperationCount> &levelOpCounts, NoiseBounds noiseBounds) {
  int ciphertextCount = levelOpCounts[moduli.size() + 1].getCiphertextCount();
  int keySwitchCount = levelOpCounts[moduli.size() + 1].getKeySwitchCount();
  int constantMulCount = levelOpCounts[moduli.size() + 1].getConstantMulCount();

  std::vector<double> bound(moduli.size() + 1);
  bound[0] = std::pow(noiseBounds.boundConstant, constantMulCount) *
            ciphertextCount * (noiseBounds.boundClean + keySwitchCount * noiseBounds.addedNoiseKeySwitching);

  std::cerr << "input bound" << bound[0] << ", log2(input bound)=" << log2(bound[0]) << std::endl;
  
  for (int i = 0; i < moduli.size(); ++i) {
    int ciphertextCount = levelOpCounts[moduli.size() - i].getCiphertextCount();
    int keySwitchCount = levelOpCounts[moduli.size() - i].getKeySwitchCount();
    int constantMulCount = levelOpCounts[moduli.size() - i].getConstantMulCount();
    
    // No multiplication for B_clean
    double a = std::pow(noiseBounds.boundConstant, constantMulCount) *
             ciphertextCount * (bound[i] * bound[i] + keySwitchCount * noiseBounds.addedNoiseKeySwitching);
    bound[i + 1] = noiseBounds.boundScale + (a / moduli[i]);

     std::cerr << "Level " << moduli.size() -i << ": bound before scale: " << a
              << ", log2(bound before scale)=" << log2(a) << std::endl
              << "modulus: " << moduli[i] << ", log2(modulus)=" << log2(moduli[i]) << std::endl
              << "bound after scale: " << bound[i + 1] << ", log2(bound)=" << log2(bound[i + 1]) << std::endl;
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

  if (getBitSize(firstMod) >= kMaxBitSize){
    return std::numeric_limits<double>::max();
  }

  return (numPrimes - 1) * log2(scalingMod) + log2(firstMod);
}

static double computeFirstModSizeFromChain(
    double p, int ringDimension, int plaintextModulus,
    const std::vector<OperationCount> &levelOpCounts, int numPrimes,
    NoiseBounds noiseBounds) {
  std::vector<double> bound =
      computeBoundChainFixed(p, numPrimes, levelOpCounts, noiseBounds);
  auto firstMod = 2 * bound.back();
  if (std::isnan(firstMod) || std::isinf(firstMod)){
    return 0;
  }
  return getBitSize(firstMod);
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
  if (getBitSize(firstMod) >= kMaxBitSize){
    return {std::numeric_limits<double>::max(), firstMod, bounds};
  }
  
  double sumModuli = 0;
  for (auto mod : moduli) {
    if (getBitSize(mod) >= kMaxBitSize){
      return {std::numeric_limits<double>::max(), firstMod, bounds};
    }
    sumModuli += log2(mod);  
  }
  
  return {sumModuli + log2(firstMod), firstMod, bounds};
}

// Forward candidate update
static std::vector<double> candidateForward(
    const std::vector<double>& moduli,
    const std::vector<OperationCount>& levelOpCounts,
    int currentIndex, double factor, int offset,
    NoiseBounds noiseBounds) {
    
  int numberModuli = moduli.size();
  
  std::vector<double> newModuli = moduli;
  newModuli[currentIndex] = factor * moduli[currentIndex];
  
  if (newModuli[currentIndex] <= 0) {
    return {};
  }
  
  auto boundsOld = computeBoundChain(moduli, levelOpCounts, noiseBounds);
  double target = boundsOld[currentIndex + offset + 1];
  
  if (target - noiseBounds.boundScale <= 0) {
    return {};
  }
  
  auto boundsTemp = computeBoundChain(newModuli, levelOpCounts, noiseBounds);
  
  double x = boundsTemp[currentIndex + offset];
  int ciphertextCount = levelOpCounts[numberModuli - (currentIndex + offset)].getCiphertextCount();
  int keySwitchCount = levelOpCounts[numberModuli - (currentIndex + offset)].getKeySwitchCount();
  int constantMulCount = levelOpCounts[numberModuli - (currentIndex + offset)].getConstantMulCount();
  
  double newPartnerModuliValue = std::pow(noiseBounds.boundConstant, constantMulCount) *
                                ciphertextCount * (x * x + keySwitchCount * noiseBounds.addedNoiseKeySwitching) /
                                (target - noiseBounds.boundScale);
  
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
  newModuli[currentIndex] = factor * moduli[currentIndex];
  
  if (newModuli[currentIndex] <= 0) {
    return {};
  }
  
  auto boundsOld = computeBoundChain(moduli, levelOpCounts, noiseBounds);

  auto bound = boundsOld[currentIndex + 1];

  for (int i = currentIndex; i > currentIndex - offset; i--) {
    int ciphertextCount = levelOpCounts[numberModuli - i].getCiphertextCount();
    int keySwitchCount = levelOpCounts[numberModuli - i].getKeySwitchCount();
    int constantMulCount = levelOpCounts[numberModuli - i].getConstantMulCount();
   
    // Solve for bound_prev: bound = B_scale + pow(B_const,cm) * n * (bound_prev^2 + k_s*B_ks) / modulus
    // → bound_prev = sqrt((bound - B_scale) * modulus / (pow(B_const,cm) * n) - k_s*B_ks)
    double factor = std::pow(noiseBounds.boundConstant, constantMulCount);
    double insideSqrt = (bound - noiseBounds.boundScale) * newModuli[i] / (factor * ciphertextCount)
                      - keySwitchCount * noiseBounds.addedNoiseKeySwitching;
    if (insideSqrt < 0) {
      return {};
    }
    bound = sqrt(insideSqrt);
  }
  
  double x = boundsOld[currentIndex - offset];
  int ciphertextCount = levelOpCounts[numberModuli - (currentIndex - offset)].getCiphertextCount();
  int keySwitchCount = levelOpCounts[numberModuli - (currentIndex - offset)].getKeySwitchCount();
  int constantMulCount = levelOpCounts[numberModuli - (currentIndex - offset)].getConstantMulCount();
  
  double newPartnerModuliValue = std::pow(noiseBounds.boundConstant, constantMulCount) *
                                ciphertextCount * (x * x + keySwitchCount * noiseBounds.addedNoiseKeySwitching) /
                                (bound - noiseBounds.boundScale);
  
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
    
  int numberModuli = moduli.size(); // Plus one due to the first modulus
  auto boundsOld = computeBoundChain(moduli, levelOpCounts, noiseBounds);
  
  double currentFirstMod = 2 * boundsOld.back();
  double newFirstMod = factor * currentFirstMod;

  auto bound = newFirstMod / 2;

  for (int i = numberModuli - 1; i > numberModuli - offset; i--) {
    int ciphertextCount = levelOpCounts[numberModuli - i].getCiphertextCount();
    int keySwitchCount = levelOpCounts[numberModuli - i].getKeySwitchCount();
    int constantMulCount = levelOpCounts[numberModuli - i].getConstantMulCount();
   
    // Solve for bound_prev: bound = B_scale + pow(B_const,cm) * n * (bound_prev^2 + k_s*B_ks) / modulus
    double factor = std::pow(noiseBounds.boundConstant, constantMulCount);
    double insideSqrt = (bound - noiseBounds.boundScale) * moduli[i] / (factor * ciphertextCount)
                      - keySwitchCount * noiseBounds.addedNoiseKeySwitching;
    if (insideSqrt < 0) {
      return {};
    }
    bound = sqrt(insideSqrt);
  }

  double x = boundsOld[numberModuli - offset];
  int ciphertextCount = levelOpCounts[offset].getCiphertextCount();
  int keySwitchCount = levelOpCounts[offset].getKeySwitchCount();
  int constantMulCount = levelOpCounts[offset].getConstantMulCount();
  
  double newPartnerModuliValue = std::pow(noiseBounds.boundConstant, constantMulCount) *
                                ciphertextCount * (x * x + keySwitchCount * noiseBounds.addedNoiseKeySwitching) /
                                (bound - noiseBounds.boundScale);
  
  if (newPartnerModuliValue <= 0) {
    return {};
  }
  
  std::vector<double> newModuli = moduli;
  newModuli[numberModuli - offset] = newPartnerModuliValue;
  
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
    
  std::vector<std::tuple<std::vector<double>, double>> candidates;
  
  // Forward updates
  for (int offset = 1; offset < moduli.size() - currentModuliIndex; ++offset) {
    auto cand = candidateForward(moduli, levelOpCounts, currentModuliIndex, factor, offset, 
                                noiseBounds);
    if (!cand.empty() && !anyModuliNegative(cand)) {
      auto [objective, firstMod, __] = computeObjective(cand, levelOpCounts, 
                                              noiseBounds);
      if (log2(firstMod) <= 0) {
        continue;
      }
      candidates.emplace_back(cand, objective);
    }
  }

  // Backward updates
    for (int offset = 1; offset <= currentModuliIndex; ++offset) {
      auto cand = candidateBackward(moduli, levelOpCounts, currentModuliIndex, factor, offset, noiseBounds);
      if (!cand.empty() && !anyModuliNegative(cand)) {
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
    int maxIter = 100, double tolerance = 0.01) {
  
  int numberModuli = levelOpCounts.size() - 1;
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
        if (candidate.empty() || anyModuliNegative(candidate)) {
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
        auto firstModSize = getBitSize(2 * bounds.back());
        if (firstModSize >= kMaxBitSize) {
          return false;
        }
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
  while (ceil(log2(pHigh)) != ceil(log2(pLow))) {
    double pMid = (pLow + pHigh) / 2.0;
    if (derivativeObjective(pMid, ringDimension, plaintextModulus,
                            levelOpCounts, numPrimes, noiseBounds) <= 0) {
      // lower then the minimizer.
      pLow = pMid;
    } else {
      // higher then the minimizer.
      pHigh = pMid;
    }
  }
  return (pLow + pHigh) / 2.0;
}

static int computeLogPQ(const std::vector<int> &moduli,
                        uint32_t numPartQOverride = 0) {
  if (moduli.empty()) {
    return 0;
  }

  // A positive override pins ω consistently across selection, validation and
  // ring-dimension computation; 0 falls back to OpenFHE's tower-count heuristic.
  auto numPartQ = (numPartQOverride > 0)
                      ? numPartQOverride
                      : ComputeNumLargeDigits(0, moduli.size() - 1);
  auto logQ = std::accumulate(moduli.begin(), moduli.end(), 0);

  int qBound = logQ;
  if (qBound != kMaxBitSize) {
    qBound += 1;
  }

  double dcrtBits = (moduli.size() > 1) ? moduli[1] : moduli[0];
  auto hybridKSInfo = lbcrypto::CryptoParametersRNS::EstimateLogP(
      numPartQ, moduli[0], dcrtBits,
      /*extraModulusSize=*/0,
      /*numPrimes=*/moduli.size(),
      /*auxBits=*/kMaxBitSize,
      /*scalTech=*/lbcrypto::FIXEDAUTO,
      /*addOne=*/true);

  auto logP = static_cast<int>(std::ceil(std::get<0>(hybridKSInfo)));
  return qBound + logP;
};

static int computeRingDimensionFromPrimes(const std::vector<int64_t> &moduli,
                                         uint32_t numPartQOverride = 0) {
  std::vector<int> moduliSizes;
  for (auto mod : moduli) {
    moduliSizes.push_back(getBitSize(mod));
  }
  auto logQP = computeLogPQ(moduliSizes, numPartQOverride);
  return lbcrypto::StdLatticeParm::FindRingDim(
      lbcrypto::HEStd_ternary, lbcrypto::HEStd_128_classic, logQP);
};

static NoiseBounds calculateBoundParams(int ringDimension, int plaintextModulus,
                                        int numPrimes,
                                        double keySwitchNoiseFactor = 1.0,
                                        uint32_t numPartQOverride = 0) {
  auto phi = ringDimension;  // Pessimistic
  auto t = plaintextModulus;
  auto d = 6.0;

  auto vKey = 2.0 / 3.0;   
  auto vErr = 3.19 * 3.19; 

  auto boundScale = d * t * sqrt((phi / 12.0) * (1.0 + (phi * vKey)));

  auto boundClean = d * t * sqrt(phi * (1.0 / 12.0 + 2 * phi * vErr * vKey + vErr));

  auto boundKeySwitch = d * t * phi * sqrt(vErr / 12.0);

  // Noise from a single constant (plaintext × ciphertext) multiplication.
  // No factor of d: d comes from the incoming bound being multiplied.
  auto boundConstant = t * sqrt(phi / 12.0);

  auto f0 = keySwitchNoiseFactor;

  // Find number of digits/partitions of Q (similar to numPartQ in OpenFHE).
  // A positive override lets the caller pin ω consistently across selection
  // (ν_KS) and validation; 0 falls back to OpenFHE's tower-count heuristic.
  auto numPartQ = (numPartQOverride > 0)
                      ? numPartQOverride
                      : ComputeNumLargeDigits(0, numPrimes - 1);
  // Calculate partitions similar to OpenFHE (ceil(sizeQ/numPartQ) towers per digit)
  int k = ceil(static_cast<double>(numPrimes) / numPartQ);

  auto addedNoiseKeySwitching = f0 * boundKeySwitch + sqrt(k) * boundScale;

  return {boundScale, boundClean, addedNoiseKeySwitching, boundConstant};
}

static int getMaxLevel(secret::GenericOp *op) {
  int maxLevel = 0;

  op->getBody()->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (op->getNumResults() == 0) {
      return;
    }
    int level = getLevelFromMgmtAttr(op->getResult(0)).getInt();
    maxLevel = std::max(maxLevel, level);
  });

  return maxLevel;
}

static std::vector<OperationCount> getLevelOpCounts(secret::GenericOp *op,
                                                    DataFlowSolver *solver,
                                                    int maxLevel) {
  std::vector<OperationCount> levelOpCounts;

  levelOpCounts.resize(maxLevel + 2, OperationCount(0, 0));
  levelOpCounts[maxLevel + 1] = OperationCount(1, 0, true);

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
    int level = getLevelFromMgmtAttr(op->getResult(0)).getInt();

    if (count.isHighestLevel()) {
      levelOpCounts[maxLevel + 1] = OperationCount::max(levelOpCounts[maxLevel + 1], count);
    } else {
      levelOpCounts[level] = OperationCount::max(levelOpCounts[level], count);
    }
  });

  return levelOpCounts;
}

static void computeModuliSizesBisection(int &firstModSize, int &scalingModSize,
                                        int ringDimension, int plaintextModulus,
                                        const std::vector<OperationCount> &levelOpCounts,
                                        int numPrimes,
                                        double keySwitchNoiseFactor = 1.0) {
  auto noiseBounds = calculateBoundParams(ringDimension, plaintextModulus,
                                          numPrimes, keySwitchNoiseFactor);

  try {
    double scalingMod = findOptimalScalingModSizeBisection(
        ringDimension, plaintextModulus, levelOpCounts, numPrimes, noiseBounds);
    
    firstModSize = computeFirstModSizeFromChain(
        scalingMod, ringDimension, plaintextModulus, levelOpCounts, numPrimes, noiseBounds);
    scalingModSize = getBitSize(scalingMod);
  } catch (const std::runtime_error& e) {
    throw; // Re-throw the exception to be caught in annotateCountParams
  }
}

static void computeModuliSizesClosed(
    int &firstModSize, int &scalingModSize, int ringDimension,
    int plaintextModulus, const std::vector<OperationCount> &levelOpCounts,
    int numPrimes, double keySwitchNoiseFactor = 1.0) {
  auto noiseBounds =
      calculateBoundParams(ringDimension, plaintextModulus, numPrimes,
                           keySwitchNoiseFactor);

  // Compute OperationCounts over all levels
  OperationCount maxCounts(0, 0);
  for (auto count : levelOpCounts) {
    maxCounts = OperationCount::max(maxCounts, count);
  }

  auto boundOptimal = noiseBounds.boundScale +
                      sqrt(noiseBounds.boundScaleSquared() +
                            (maxCounts.getKeySwitchCount() *
                            noiseBounds.addedNoiseKeySwitching));

  auto boundStart = std::pow(noiseBounds.boundConstant, maxCounts.getConstantMulCount()) *
            maxCounts.getCiphertextCount() *
            (noiseBounds.boundClean +
              maxCounts.getKeySwitchCount() * noiseBounds.addedNoiseKeySwitching);

  if (boundOptimal >= boundStart) {
    auto scalingMod = std::pow(noiseBounds.boundConstant, maxCounts.getConstantMulCount()) *
                      2 * maxCounts.getCiphertextCount() * boundOptimal;
    scalingModSize = getBitSize(scalingMod);
    firstModSize = getBitSize(2 * boundOptimal);
  } else {
    auto scalingMod = (std::pow(noiseBounds.boundConstant, maxCounts.getConstantMulCount()) *
                      maxCounts.getCiphertextCount() * (boundStart * boundStart + maxCounts.getKeySwitchCount() * noiseBounds.addedNoiseKeySwitching)) /
                      (boundStart - noiseBounds.boundScale);
    scalingModSize = getBitSize(scalingMod);
    firstModSize = getBitSize(2 * boundStart);
  }
}

static std::vector<int> computeModuliSizesBalancing(
    int ringDimension, int plaintextModulus,
    const std::vector<OperationCount> &levelOpCounts, int numPrimes,
    double keySwitchNoiseFactor = 1.0) {
  auto noiseBounds =
      calculateBoundParams(ringDimension, plaintextModulus, numPrimes,
                           keySwitchNoiseFactor);

  // Use bisection result as init value
  double pInit = findOptimalScalingModSizeBisection(
    ringDimension, plaintextModulus, levelOpCounts, numPrimes, noiseBounds);

  auto rebalanced = rebalancingModuli(pInit, levelOpCounts, noiseBounds);
  
  std::vector<int> moduli;

  moduli.reserve(rebalanced.size());
  for (const auto& p : rebalanced) {
    moduli.push_back(getBitSize(p));
  }

  auto sumLast = moduli[0] + moduli[1];
  moduli[0] = ceil(sumLast / 2);
  moduli[1] = sumLast - moduli[0];
  
  return moduli;
}

static std::vector<int> computeModuliSizesGreedy(
    int ringDimension, int plaintextModulus,
    const std::vector<OperationCount>& levelOpCounts, int numPrimes,
    double keySwitchNoiseFactor = 1.0,
    uint32_t numPartQOverride = 0) {
  auto noiseBounds =
      calculateBoundParams(ringDimension, plaintextModulus, numPrimes,
                           keySwitchNoiseFactor, numPartQOverride);

  // Print the level operation counts and noise bounds for debugging
  std::cerr << "Level operation counts:" << std::endl;
  for (size_t i = 0; i < levelOpCounts.size(); ++i) {
    std::cerr << "Level " << i
              << ": CiphertextCount=" << levelOpCounts[i].getCiphertextCount()
              << ", KeySwitchCount=" << levelOpCounts[i].getKeySwitchCount()
              << std::endl;
  }

  int numScalingModuli = numPrimes - 1;
  std::vector<double> scalingModuli;
  scalingModuli.reserve(numScalingModuli);

  std::cerr << "Starting greedy modulus selection with noise bounds: "
            << "boundScale=" << noiseBounds.boundScale
            << ", log2(boundScale)=" << log2(noiseBounds.boundScale)
            << ", boundClean=" << noiseBounds.boundClean
            << ", log2(boundClean)=" << log2(noiseBounds.boundClean)
            << ", addedNoiseKeySwitching=" << noiseBounds.addedNoiseKeySwitching
            << ", log2(addedNoiseKeySwitching)="
            << log2(noiseBounds.addedNoiseKeySwitching) << std::endl;

  int startCiphertextCount =
      levelOpCounts[numScalingModuli + 1].getCiphertextCount();
  int startKeySwitchCount =
      levelOpCounts[numScalingModuli + 1].getKeySwitchCount();
  int startConstantMulCount =
      levelOpCounts[numScalingModuli + 1].getConstantMulCount();
  double currentBound =
      std::pow(noiseBounds.boundConstant, startConstantMulCount) *
      startCiphertextCount *
      (noiseBounds.boundClean +
       startKeySwitchCount * noiseBounds.addedNoiseKeySwitching);

  std::cerr << "Initial noise bound before scaling moduli: " << currentBound
            << ", log2(currentBound)=" << log2(currentBound) << std::endl;

  for (int i = 0; i < numScalingModuli; ++i) {
    int levelIndex = numScalingModuli - i;
    int ciphertextCount = levelOpCounts[levelIndex].getCiphertextCount();
    int keySwitchCount = levelOpCounts[levelIndex].getKeySwitchCount();
    int constantMulCount = levelOpCounts[levelIndex].getConstantMulCount();

    // The incoming noise is squared only when a ct×ct multiplication
    // occurred at the previous (higher) level.  A ct×ct mul resets
    // ciphertextCount to 1, so ciphertextCount > 1 at the current level
    // signals that only additions / plain multiplications happened (noise
    // grows linearly, not quadratically).
    double incomingBound =
        (ciphertextCount > 1) ? currentBound : (currentBound * currentBound);

    double levelNoiseTerm =
        std::pow(noiseBounds.boundConstant, constantMulCount) *
        ciphertextCount * (incomingBound +
                           keySwitchCount * noiseBounds.addedNoiseKeySwitching);

    double greedyQi = 2.0 * levelNoiseTerm / noiseBounds.boundScale;
  
    if (getBitSize(greedyQi) >= kMaxBitSize) {
      greedyQi = std::pow(2.0, kMaxBitSize - 1);
    }

    scalingModuli.push_back(greedyQi);
    currentBound = noiseBounds.boundScale + (levelNoiseTerm / greedyQi);

    std::cerr << "Level " << levelIndex << " selected modulus: " << greedyQi
          << " (bit size: " << getBitSize(greedyQi) << ")" 
          << ", bound before scale: " << levelNoiseTerm << ", log2(bound before scale)=" << log2(levelNoiseTerm)
          << ",  bound after scale: " << currentBound
          << ", log2(bound after scale)=" << log2(currentBound) << std::endl;
  }

  double firstMod = 2.0 * currentBound;
  
  if (getBitSize(firstMod) >= kMaxBitSize) {
    throw std::runtime_error(
        "Greedy selection infeasible: first modulus exceeds maximum bit size");
  }

  std::vector<int> moduli;
  moduli.reserve(numPrimes);
  moduli.push_back(getBitSize(firstMod));
  for (auto it = scalingModuli.rbegin(); it != scalingModuli.rend(); ++it) {
    moduli.push_back(getBitSize(*it));
  }

  return moduli;
}

// Print parameters directly to std::cerr with result tags
void printParamsWithResultTags(const std::vector<int> &moduli, int ringDimension, 
                               int plaintextModulus, const std::string &testname,
                               const std::string &selectionApproach) {
  // Generate Unix timestamp in seconds
  auto now = std::chrono::system_clock::now();
  auto unix_timestamp =
      std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch())
          .count();

  auto toLowerCase = [](std::string str) {
    std::transform(str.begin(), str.end(), str.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return str;
  };

  // Generate JSON without using CryptoContext
  std::stringstream ss;
  ss << "{\n";
  ss << R"(  "testname": ")" << testname << "\",\n";
  ss << R"(  "selectionApproach": ")" << toLowerCase(selectionApproach) << "\",\n";
  ss << R"(  "timestamp": )" << unix_timestamp << ",\n";
  ss << "  \"modulusSizes\": [";
  for (size_t i = 0; i < moduli.size(); ++i) {
    ss << moduli[i];
    if (i != moduli.size() - 1) {
      ss << ", ";
    }
  }
  ss << "],\n";
  ss << "  \"totalSize\": " << std::accumulate(moduli.begin(), moduli.end(), 0) << ",\n";
  ss << "  \"ringDimension\": " << ringDimension << ",\n";
  ss << "  \"plaintextModulus\": " << plaintextModulus << "\n";
  ss << "}";

  std::cerr << "<params>" << ss.str() << "</params>" << std::endl;
}

using BigInteger = bigintbackend::BigInteger;

static bool validateHybridAssumptionFromSizes(const std::vector<int> &moduliSizes,
                                              int auxBits,
                                              uint32_t numPartQOverride = 0) {
  if (moduliSizes.empty()) {
    return false;
  }

  auto numPartQ = (numPartQOverride > 0)
                      ? numPartQOverride
                      : ComputeNumLargeDigits(0, moduliSizes.size() - 1);
  if (numPartQ == 0) {
    return false;
  }

  uint32_t towersPerPart =
      ceil(static_cast<double>(moduliSizes.size()) / numPartQ);

  uint32_t maxBits = 0;
  for (uint32_t j = 0; j < numPartQ; ++j) {
    uint32_t partBits = 0;
    for (uint32_t i = towersPerPart * j; i < (j + 1) * towersPerPart; ++i) {
      if (i < moduliSizes.size()) {
        partBits += moduliSizes[i];
      }
    }
    maxBits = std::max(maxBits, partBits);
  }

  uint32_t sizeP = ceil(static_cast<double>(maxBits) / auxBits);
  if (sizeP == 0) {
    return false;
  }

  // Diagnostic: trace the hybrid key-switching assumption check
  double lhs = log2(sqrt(static_cast<double>(numPartQ) * moduliSizes.size())) +
               maxBits;
  double rhs = static_cast<double>(auxBits) * sizeP;
  std::cerr << "Hybrid assumption check: numPartQ=" << numPartQ
            << ", towersPerPart=" << towersPerPart
            << ", maxBits=" << maxBits
            << ", sizeP=" << sizeP
            << ", lhs=" << lhs
            << ", rhs=" << rhs
            << (lhs <= rhs ? " [PASS]" : " [FAIL]") << std::endl;

  return lhs <= rhs;
}

static bool validateHybridAssumptionFromPrimes(const std::vector<int64_t> &moduli,
                                              int auxBits,
                                              uint32_t numPartQOverride = 0) {
  std::vector<int> moduliSizes;
  for (auto mod : moduli) {
    moduliSizes.push_back(getBitSize(mod));
  }
  return validateHybridAssumptionFromSizes(moduliSizes, auxBits, numPartQOverride);
}

static std::vector<int64_t> computePiModuli(const std::vector<int64_t> &qi,
                                            int ringDimension,
                                            int plaintextModulus,
                                            uint32_t numPartQOverride = 0) {
  // Following OpenFHE's approach for extension moduli in HYBRID key switching
  std::vector<int64_t> pi;

  // Calculate auxiliary primes bit size (auxBits)
  // In OpenFHE, this is typically set to 60 bits for 128-bit security
  int auxBits = heir::kMaxBitSize;

  // Find number of digits/partitions of Q (similar to numPartQ in OpenFHE)
  auto numPartQ = (numPartQOverride > 0)
                      ? numPartQOverride
                      : ComputeNumLargeDigits(0, qi.size() - 1);

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
    if (bits > maxBits) {
      maxBits = bits;
    }
  }

  // Select number of primes in auxiliary CRT basis
  uint32_t sizeP = ceil(static_cast<double>(maxBits) / auxBits);

  // Start with first prime as done in OpenFHE
  lbcrypto::NativeInteger firstP =
      lbcrypto::FirstPrime<lbcrypto::NativeInteger>(auxBits, 2 * ringDimension);
  lbcrypto::NativeInteger pPrev = firstP;

  // Generate each auxiliary prime
  for (uint32_t i = 0; i < sizeP; i++) {
    // The following loop makes sure that moduli in P and Q are different
    lbcrypto::NativeInteger currentP;
    bool foundInQ;
    do {
      currentP =
          lbcrypto::PreviousPrime<lbcrypto::NativeInteger>(pPrev, 2 * ringDimension);
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

// On hybrid-key-switching-assumption failure, grow the moduli of a *sub-maximal*
// digit so that it becomes the new maximum and crosses the next auxBits boundary.
// This increases sizeP (hence P), restoring the slack the check needs, while only
// *enlarging* moduli (which never breaks correctness — larger q_i only add noise
// margin). The selected moduli are minimal-but-not-fixed: if they must be larger
// for the key-switching validation to hold, we may grow them.
// Returns the grown bit-sizes, or empty if no sub-maximal digit can be grown
// across the boundary (e.g. all its moduli are already at the cap).
static std::vector<int> growDigitForHybridAssumption(
    const std::vector<int> &moduliSizes, uint32_t numPartQ,
    int auxBits = kMaxBitSize) {
  if (moduliSizes.empty() || numPartQ == 0) return {};
  int n = static_cast<int>(moduliSizes.size());
  uint32_t towersPerPart =
      static_cast<uint32_t>(std::ceil(static_cast<double>(n) / numPartQ));

  // Digit bit-sums and tower ranges (uniform consecutive, matching OpenFHE).
  std::vector<uint32_t> digitBits(numPartQ, 0);
  std::vector<std::pair<int, int>> digitRange(numPartQ, {0, 0});
  for (uint32_t j = 0; j < numPartQ; ++j) {
    int start = static_cast<int>(towersPerPart * j);
    int end = std::min(static_cast<int>(towersPerPart * (j + 1)), n);
    digitRange[j] = {start, end};
    for (int i = start; i < end; ++i) digitBits[j] += moduliSizes[i];
  }

  uint32_t maxDigit = 0;
  for (uint32_t j = 1; j < numPartQ; ++j)
    if (digitBits[j] > digitBits[maxDigit]) maxDigit = j;

  uint32_t maxBits = digitBits[maxDigit];
  uint32_t sizeP = static_cast<uint32_t>(
      std::ceil(static_cast<double>(maxBits) / auxBits));
  int pBits = auxBits * static_cast<int>(sizeP);
  double slackNeeded =
      log2(sqrt(static_cast<double>(numPartQ) * moduliSizes.size()));

  // Already passing: nothing to do.
  if (static_cast<double>(maxBits) + slackNeeded <= pBits) return moduliSizes;

  // Try each sub-maximal digit: grow its non-capped moduli so the digit crosses
  // the current P boundary (pBits), making it the new max and bumping sizeP.
  for (uint32_t j = 0; j < numPartQ; ++j) {
    if (j == maxDigit) continue;
    int growable = 0;
    for (int i = digitRange[j].first; i < digitRange[j].second; ++i)
      if (moduliSizes[i] < auxBits) growable += auxBits - moduliSizes[i];
    if (growable == 0) continue;  // digit fully capped

    int target = pBits + 1 - static_cast<int>(digitBits[j]);  // bits to add
    if (target <= 0) continue;  // already past boundary (not the non-max)
    if (target > growable) continue;  // can't grow enough

    std::vector<int> grown = moduliSizes;
    int toAdd = target + 1;  // +1 for strict crossing
    for (int i = digitRange[j].first; i < digitRange[j].second && toAdd > 0; ++i) {
      int room = auxBits - grown[i];
      int add = std::min(room, toAdd);
      grown[i] += add;
      toAdd -= add;
    }
    if (toAdd > 0) continue;
    return grown;
  }
  return {};
}

static std::vector<int64_t> selectLattigoPrimesFromSizes(
    const std::vector<int> &moduliSizes, int ringDimension,
    int plaintextModulus) {
  if (moduliSizes.empty()) {
    return {};
  }

  uint64_t modulusOrder =
      computeModulusOrder(ringDimension, plaintextModulus);
  std::vector<int64_t> selectedPrimes;
  selectedPrimes.reserve(moduliSizes.size());

  for (int requestedSize : moduliSizes) {
    if (requestedSize > kMaxBitSize) {
      throw std::runtime_error("Requested modulus size exceeds maximum bit size");
    }

    std::cerr << "Selecting prime of size " << requestedSize
              << " bits for modulus with order " << modulusOrder << std::endl;

    auto currentPrime = lbcrypto::FirstPrime<lbcrypto::NativeInteger>(
        requestedSize - 1, modulusOrder);

    while (std::any_of(selectedPrimes.begin(), selectedPrimes.end(),
                       [&](int64_t existingPrime) {
                         return existingPrime ==
                                static_cast<int64_t>(currentPrime.ConvertToInt());
                       })) {
      currentPrime = lbcrypto::NextPrime<lbcrypto::NativeInteger>(currentPrime, modulusOrder);
    }

    selectedPrimes.push_back(
        static_cast<int64_t>(currentPrime.ConvertToInt()));
  }

  return selectedPrimes;
}

static void annotateSchemeParam(Operation *op, const uint64_t plaintextModulus,
                         const uint64_t ringDimension, const std::vector<int64_t>& moduli,
                         uint32_t numPartQOverride = 0) {
  std::vector<int64_t> qi = moduli;

  // Compute pi moduli (extension moduli)
  std::vector<int64_t> pi =
      computePiModuli(qi, ringDimension, plaintextModulus, numPartQOverride);

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
                                  int firstModSize, int scalingModSize,
                  int plaintextModulus,
                  OperationCount maxCounts) {
  auto *funcOp = ((Operation*) genericOp)->getParentOp();

  auto openfheParamAttr = mgmt::OpenfheParamsAttr::get(
    funcOp->getContext(),
    maxCounts.getCiphertextCount(),
    firstModSize,
    maxCounts.getKeySwitchCount(),
    multiplicativeDepth,
    plaintextModulus,
    ringDimension,
    scalingModSize
  );

  funcOp->setAttr(mgmt::MgmtDialect::kArgOpenfheParamsAttrName,
          openfheParamAttr);
}

static std::pair<int, int> findValidOpenFhePrimeSizes(
    int computedFirstModSize, int computedScalingModSize, int numPrimes,
    int ringDimension, int plaintextModulus,
    std::function<int(int)> recomputeFirstModSize) {
  uint64_t modulusOrder = computeModulusOrder(ringDimension, plaintextModulus);
  lbcrypto::NativeInteger firstMod = 0;

  auto firstModSize = computedFirstModSize;
  auto scalingModSize = computedScalingModSize;

  while (scalingModSize < kMaxBitSize && firstModSize < kMaxBitSize) {
    try {
      firstMod = lbcrypto::LastPrime<lbcrypto::NativeInteger>(
          firstModSize, modulusOrder);
    } catch (lbcrypto::OpenFHEException &) {
      firstModSize += 1;
      continue;
    }

    if (getBitSize(firstMod.ConvertToInt()) < computedFirstModSize) {
        firstModSize += 1;
        continue;
    }

    lbcrypto::NativeInteger q;
    if (firstModSize == scalingModSize) {
      q = firstMod;
    } else {
      q = lbcrypto::LastPrime<lbcrypto::NativeInteger>(scalingModSize,
                                                        modulusOrder);
    }
    bool allFound = true;
    try {
      for (int i = 1; i < numPrimes; i++) {
        q = lbcrypto::PreviousPrime<lbcrypto::NativeInteger>(q, modulusOrder);
        if (q == firstMod || getBitSize(q.ConvertToInt()) < computedScalingModSize) {
          allFound = false;
          break;
        }
      }
      if (allFound || numPrimes == 1) {
        break;
      }
    } catch (lbcrypto::OpenFHEException &) {
      allFound = false;
    }

    if (!allFound) {
      if (firstModSize == scalingModSize) {
        firstModSize += 1;
      } else {
        scalingModSize += 1;
        firstModSize = recomputeFirstModSize(scalingModSize);
      }
    }
  }

  if (scalingModSize >= kMaxBitSize || firstModSize >= kMaxBitSize) {
    throw std::runtime_error(
      "OpenFHE prime validation failed: could not find valid prime sizes.");
  }

  return {firstModSize, scalingModSize};
}

static int computeRingDimensionFromOpenfheSizes(int firstModSize,
                                                int scalingModSize,
                                                int numPrimes) {
  auto numPartQ = ComputeNumLargeDigits(0, numPrimes - 1);
  auto logQ = firstModSize + (numPrimes - 1) * scalingModSize;

  if (logQ != kMaxBitSize) {
    logQ += 1;
  }

  double dcrtBits = (numPrimes > 1) ? scalingModSize : firstModSize;
  auto hybridKSInfo = lbcrypto::CryptoParametersRNS::EstimateLogP(
      numPartQ, firstModSize, dcrtBits,
      /*extraModulusSize=*/0,
      /*numPrimes=*/numPrimes,
      /*auxBits=*/kMaxBitSize,
      /*scalTech=*/lbcrypto::FIXEDAUTO,
      /*addOne=*/true);

  auto logP = static_cast<int>(std::ceil(std::get<0>(hybridKSInfo)));
  auto logQP = logQ + logP;

  return lbcrypto::StdLatticeParm::FindRingDim(
      lbcrypto::HEStd_ternary, lbcrypto::HEStd_128_classic, logQP);
}

static std::vector<int64_t> findValidPrimesLattigo(
    const std::vector<int> &computedModuliSizes, int ringDimension,
    int plaintextModulus) {
  auto selectedPrimes =
      selectLattigoPrimesFromSizes(computedModuliSizes, ringDimension,
                                   plaintextModulus);

  return selectedPrimes;
}

void annotateCountParams(Operation *top, DataFlowSolver *solver,
                         int ringDimension, int plaintextModulus,
                         std::string algorithm) {
  top->walk<WalkOrder::PreOrder>([&](secret::GenericOp genericOp) {
    if (algorithm != "DIRECT" && algorithm != "CLOSED" &&
        algorithm != "BISECTION" && algorithm != "GREEDY" &&
        algorithm != "BALANCING") {
      genericOp->emitOpError()
          << "Unsupported algorithm '" << algorithm
          << "'. Supported values are: DIRECT, CLOSED, BISECTION, GREEDY, BALANCING.";
      return;
    }

    bool isRingDimensionSet = ringDimension != 0;

    auto maxLevel = getMaxLevel(&genericOp);
    auto levelOpCounts = getLevelOpCounts(&genericOp, solver, maxLevel);

    OperationCount maxCounts(0, 0);
    for (auto count : levelOpCounts) {
      maxCounts = OperationCount::max(maxCounts, count);
    }

    auto multiplicativeDepth = maxLevel;
    auto numPrimes = multiplicativeDepth + 1;

    if (algorithm == "DIRECT") {
      annotateOpenfheParams(genericOp, multiplicativeDepth, ringDimension,
                            0, 0, plaintextModulus, maxCounts);
      return;
    }

    const int initialRingDimension = isRingDimensionSet ? ringDimension : 16384;

    double keySwitchNoiseFactor = 1.0;
    int validatedFirstModSize = 0;
    int validatedScalingModSize = 0;

    // Hybrid key-switching digit count (ω). The paper (§4.5) prescribes the
    // smallest ω ≥ 2 such that PQ maintains the target security; OpenFHE's
    // ComputeNumLargeDigits table is only a fallback. For GREEDY we start at ω=2
    // and escalate only if recovery fails; other algorithms keep the table value.
    uint32_t currentOmega = 0;
    if (algorithm == "GREEDY") currentOmega = 2;

    // Floor on modulus bit-sizes carried across retries: when the hybrid check
    // fails because the max digit sits on an auxBits boundary, we grow a
    // sub-maximal digit's moduli and enforce those sizes as a minimum on the
    // next selection pass. Empty means "use greedy-minimal sizes".
    std::vector<int> moduliSizeFloor;
    int growAttempts = 0;
    const int kMaxGrowAttempts = 4;

    auto applyFloor = [&](std::vector<int> sizes) {
      if (moduliSizeFloor.size() == sizes.size()) {
        for (size_t i = 0; i < sizes.size(); ++i)
          sizes[i] = std::max(sizes[i], moduliSizeFloor[i]);
      }
      return sizes;
    };

    auto computeModuliSizes =
        [&](int ringDimension) -> std::vector<int> {
      if (numPrimes == 1) {
        auto noiseBounds = calculateBoundParams(
            ringDimension, plaintextModulus, numPrimes, keySwitchNoiseFactor,
            currentOmega);
        int firstModSize = floor(1 + log2(std::pow(noiseBounds.boundConstant, levelOpCounts[1].getConstantMulCount()) *
                            levelOpCounts[1].getCiphertextCount() * (noiseBounds.boundClean + levelOpCounts[1].getKeySwitchCount() * noiseBounds.addedNoiseKeySwitching))) + 1;
        return applyFloor({firstModSize});
      }
      try {
        if (algorithm == "BALANCING") {
          return applyFloor(computeModuliSizesBalancing(ringDimension, plaintextModulus,
                                             levelOpCounts, numPrimes,
                                             keySwitchNoiseFactor));
        }
        if (algorithm == "GREEDY") {
          return applyFloor(computeModuliSizesGreedy(ringDimension, plaintextModulus,
                                          levelOpCounts, numPrimes,
                                          keySwitchNoiseFactor, currentOmega));
        }
      } catch (const std::runtime_error& e) {
        genericOp->emitOpError() << "Parameter optimization failed: " << e.what();
        return {};
      }
      return {};
    };

    auto computeValidatedOpenfheSizes = [&](int candidateRingDimension,
                                            int &outFirstModSize,
                                            int &outScalingModSize) -> bool {
      int computedFirstModSize = 0;
      int computedScalingModSize = 0;
      try {
        if (algorithm == "BISECTION") {
          computeModuliSizesBisection(computedFirstModSize, computedScalingModSize,
                                      candidateRingDimension, plaintextModulus,
                                      levelOpCounts, numPrimes,
                                      keySwitchNoiseFactor);
        } else {
          computeModuliSizesClosed(computedFirstModSize, computedScalingModSize,
                                   candidateRingDimension, plaintextModulus,
                                   levelOpCounts, numPrimes,
                                   keySwitchNoiseFactor);
        }

        std::function<int(int)> recomputeFirstModSize;
        if (algorithm == "BISECTION") {
          recomputeFirstModSize = [&](int currentScalingModSize) -> int {
            double scalingMod = pow(2.0, currentScalingModSize);
            auto noiseBounds = calculateBoundParams(
                candidateRingDimension, plaintextModulus, numPrimes,
                keySwitchNoiseFactor);
            return computeFirstModSizeFromChain(
                scalingMod, candidateRingDimension, plaintextModulus,
                levelOpCounts, numPrimes, noiseBounds);
          };
        } else {
          recomputeFirstModSize =
              [computedFirstModSize](int) { return computedFirstModSize; };
        }

        auto validatedSizes = findValidOpenFhePrimeSizes(
            computedFirstModSize, computedScalingModSize, numPrimes,
            candidateRingDimension, plaintextModulus, recomputeFirstModSize);
        outFirstModSize = validatedSizes.first;
        outScalingModSize = validatedSizes.second;
        return true;
      } catch (const std::runtime_error &e) {
        genericOp->emitOpError() << e.what();
        return false;
      }
    };

    std::vector<int64_t> moduli;

    while (true) {
      ringDimension = initialRingDimension;
      int newRingDimension = ringDimension;
      
      moduli.clear();

      validatedFirstModSize = 0;
      validatedScalingModSize = 0;

      while (true) {
        if (algorithm == "BISECTION" || algorithm == "CLOSED") {
          if (!computeValidatedOpenfheSizes(
                  ringDimension, validatedFirstModSize,
                  validatedScalingModSize)) {
            return;
          }

          if (isRingDimensionSet) {
            break;
          }

          newRingDimension = computeRingDimensionFromOpenfheSizes(
              validatedFirstModSize, validatedScalingModSize, numPrimes);
          
          if (newRingDimension == ringDimension) {
            int smallerDimension = ringDimension / 2;
            int smallerFirstModSize = 0;
            int smallerScalingModSize = 0;
            if (!computeValidatedOpenfheSizes(
                    smallerDimension, smallerFirstModSize,
                      smallerScalingModSize)) {
              break;
            }

            newRingDimension = computeRingDimensionFromOpenfheSizes(
                smallerFirstModSize, smallerScalingModSize, numPrimes);

            if (newRingDimension == smallerDimension) {
              ringDimension = smallerDimension;
              validatedFirstModSize = smallerFirstModSize;
              validatedScalingModSize = smallerScalingModSize;
            } else {
              break;
            }
          } else {
            ringDimension = newRingDimension;
          }
        } else {
          auto computedModuliSizes = computeModuliSizes(ringDimension);
          if (computedModuliSizes.empty()) {
            break;
          }

          try {
            moduli = findValidPrimesLattigo(computedModuliSizes, ringDimension,
                                            plaintextModulus);
          } catch (const std::runtime_error &e) {
            genericOp->emitOpError()
                << "Prime validation failed: " << e.what();
            moduli.clear();
            break;
          }

          if (moduli.empty()) {
            break;
          }

          if (isRingDimensionSet) {
            break;
          }

          newRingDimension = computeRingDimensionFromPrimes(moduli,
                                                              currentOmega);

          if (newRingDimension == ringDimension) {
            // Try smaller ring dimension.
            int smallerDimension = ringDimension / 2;

            auto smallerComputedModuliSizes =
                computeModuliSizes(smallerDimension);

            if (smallerComputedModuliSizes.empty()) {
              break;
            }

            std::vector<int64_t> smallerValidatedModuli;
            try {
              smallerValidatedModuli =
                  findValidPrimesLattigo(smallerComputedModuliSizes,
                                         smallerDimension, plaintextModulus);
            } catch (const std::runtime_error &) {
              break;
            }

            if (smallerValidatedModuli.empty()) {
              break;
            }

            newRingDimension = computeRingDimensionFromPrimes(smallerValidatedModuli,
                                                                currentOmega);

            if (newRingDimension == smallerDimension) {
              ringDimension = smallerDimension;
              moduli = smallerValidatedModuli;
            } else {
              break;
            }
          } else {
            ringDimension = newRingDimension;
          }
        }
      }

      if (algorithm == "GREEDY" || algorithm == "BALANCING") {
        if (moduli.empty()) {
          throw std::runtime_error("Failed to find valid modulus sizes.");
          return;
        }

        if (validateHybridAssumptionFromPrimes(moduli, kMaxBitSize,
                                              currentOmega)) {
          annotateSchemeParam(top, plaintextModulus, ringDimension, moduli,
                              currentOmega);
          // Convert primes to bit sizes for OpenFHE params and reporting
          std::vector<int> moduliSizes;
          moduliSizes.reserve(moduli.size());
          for (auto prime : moduli) {
            moduliSizes.push_back(
                static_cast<int>(std::floor(std::log2((double)prime)) + 1));
          }
          int firstModSize = moduliSizes[0];
          int scalingModSize =
              *std::max_element(moduliSizes.begin() + 1, moduliSizes.end());
          annotateOpenfheParams(genericOp, multiplicativeDepth, ringDimension,
                                firstModSize, scalingModSize, plaintextModulus,
                                maxCounts);
          printParamsWithResultTags(moduliSizes, ringDimension, plaintextModulus,
                                    "<testname>", algorithm);
          return;
        }

        // Hybrid key-switching assumption failed. The selected moduli are minimal
        // but not fixed: if enlarging a sub-maximal digit's moduli lets the max
        // digit cross the next auxBits boundary, sizeP (hence P) grows and the
        // check recovers. We enforce the grown sizes as a floor on the next pass.
        if (algorithm == "GREEDY") {
          std::vector<int> bitSizes;
          bitSizes.reserve(moduli.size());
          for (auto prime : moduli)
            bitSizes.push_back(
                static_cast<int>(std::floor(std::log2((double)prime)) + 1));
          auto grown = growDigitForHybridAssumption(bitSizes, currentOmega);
          if (!grown.empty() && growAttempts < kMaxGrowAttempts) {
            moduliSizeFloor = grown;
            growAttempts++;
            continue;
          }
          // Grow exhausted: escalate ω to the OpenFHE table value, then declare
          // the circuit infeasible for greedy selection.
          if (currentOmega == 2) {
            currentOmega = ComputeNumLargeDigits(0, numPrimes - 1);
            moduliSizeFloor.clear();
            growAttempts = 0;
            continue;
          }
          genericOp->emitOpError()
              << "Greedy parameter selection infeasible: hybrid key-switching "
                 "assumption cannot be satisfied (the largest digit saturates "
                 "the modulus cap). No valid parameter set exists for this circuit.";
          return;
        }

        // BALANCING: bounded keySwitchNoiseFactor retry, then infeasible.
        if (keySwitchNoiseFactor < 5.0) {
          keySwitchNoiseFactor += 0.1;
          continue;
        }
        genericOp->emitOpError()
            << "Balancing parameter selection infeasible: hybrid key-switching "
               "assumption cannot be satisfied.";
        return;
      }

      if (algorithm == "BISECTION" || algorithm == "CLOSED") {
        if (validatedFirstModSize == 0 || validatedScalingModSize == 0) {
          genericOp->emitOpError() << "Unable to derive valid OpenFHE modulus sizes.";
          return;
        }

        std::vector<int> openfheModuliSizes;
        openfheModuliSizes.reserve(numPrimes);
        openfheModuliSizes.push_back(validatedFirstModSize);
        for (int i = 1; i < numPrimes; ++i) {
          openfheModuliSizes.push_back(validatedScalingModSize);
        }
        if (!validateHybridAssumptionFromSizes(openfheModuliSizes, kMaxBitSize)) {
          if (keySwitchNoiseFactor < 5.0) {
            keySwitchNoiseFactor += 0.1;
            continue;
          }
          genericOp->emitOpError()
              << "Parameter selection infeasible: hybrid key-switching "
                 "assumption cannot be satisfied.";
          return;
        }

        // Compute actual primes from bit sizes for bgv.scheme_param
        std::vector<int64_t> moduliPrimes;
        try {
          moduliPrimes = findValidPrimesLattigo(openfheModuliSizes,
                                                ringDimension, plaintextModulus);
        } catch (const std::runtime_error &e) {
          genericOp->emitOpError()
              << "Prime computation failed: " << e.what();
          return;
        }
        annotateSchemeParam(top, plaintextModulus, ringDimension, moduliPrimes);
        annotateOpenfheParams(genericOp, multiplicativeDepth, ringDimension,
                              validatedFirstModSize, validatedScalingModSize,
                              plaintextModulus, maxCounts);
        printParamsWithResultTags(openfheModuliSizes, ringDimension,
                                 plaintextModulus, "<testname>", algorithm);
        return;
      }
      break;
    }

  });
}
}  // namespace heir
}  // namespace mlir