#include "lib/Parameters/RLWEParams.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <ios>
#include <numeric>
#include <sstream>
#include <vector>

#include "lib/Parameters/RLWESecurityParams.h"
#include "lib/Utils/Utils.h"
#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project
#include "src/core/include/openfhecore.h"           // from @openfhe

namespace mlir {
namespace heir {

// from OpenFHE, the empirical way to select dnum based on level
int computeDnum(int level) {
  if (level > 3) {
    return 3;
  }
  if (level > 0) {
    return 2;
  }
  return 1;
}

RLWESchemeParam RLWESchemeParam::getConservativeRLWESchemeParam(
    int level, int minRingDim, bool usePublicKey,
    bool encryptionTechniqueExtended) {
  auto logModuli = 60;  // assume all 60 bit moduli
  auto dnum = computeDnum(level);
  std::vector<double> logqi(level + 1, logModuli);
  std::vector<double> logpi(ceil(static_cast<double>(logqi.size()) / dnum),
                            logModuli);

  auto totalQP = logModuli * (logqi.size() + logpi.size());

  auto ringDim = computeRingDim(totalQP, minRingDim);

  return RLWESchemeParam(ringDim, level, logqi, dnum, logpi, usePublicKey,
                         encryptionTechniqueExtended);
}

int64_t findPrime(int qi, int ringDim,
                  const std::vector<int64_t>& existingPrimes) {
  std::cerr << "[findPrime] Starting search | ringDim=" << ringDim
            << ", initial qi=" << qi << "\n";

  while (qi < 80) {
    try {
      // openfhe FirstPrime will throw exception if it fails to find a prime
      bool redo = false;
      int64_t dupPrime;
      do {
        int64_t prime;
        if (!redo) {
          // first time, use first prime
          auto res =
              lbcrypto::FirstPrime<lbcrypto::NativeInteger>(qi, 2 * ringDim);
          prime = res.ConvertToInt();
          std::cerr << "[findPrime] FirstPrime candidate: " << prime
                    << " (qi=" << qi << ")\n";
        } else {
          // start from the duplicated prime
          auto res = lbcrypto::NextPrime(lbcrypto::NativeInteger(dupPrime),
                                         2 * ringDim);
          prime = res.ConvertToInt();
          std::cerr << "[findPrime] NextPrime candidate: " << prime
                    << " (after duplicate " << dupPrime << ")\n";
        }
        if (std::find(existingPrimes.begin(), existingPrimes.end(), prime) ==
            existingPrimes.end()) {
              std::cerr << "[findPrime] Accepted prime: " << prime << "\n";
          return prime;
        }
        std::cerr << "[findPrime] Prime " << prime
                  << " already exists, trying next...\n";
        dupPrime = prime;
        redo = true;
      } while (redo);
    } catch (...) {
      std::cerr << "[findPrime] FirstPrime failed at qi=" << qi
                << ", retrying with qi=" << (qi + 1) << "\n";
      qi += 1;
    }
  }
  assert(false && "failed to generate good qi");
  return 0;
}

RLWESchemeParam RLWESchemeParam::getConcreteRLWESchemeParam(
    std::vector<double> logqi, int minRingDim, bool usePublicKey,
    bool encryptionTechniqueExtended, int64_t plaintextModulus) {
  auto level = logqi.size() - 1;
  auto dnum = computeDnum(level);

  std::cerr << "[getConcreteRLWESchemeParam] Starting | level=" << level
            << ", dnum=" << dnum << ", minRingDim=" << minRingDim
            << ", plaintextModulus=" << plaintextModulus << "\n";

  // sanitize qi
  for (auto& qi : logqi) {
    if (qi < 20) {
      std::cerr << "[getConcreteRLWESchemeParam] Sanitizing qi: " << qi
                << " -> 20\n";
      qi = 20;
    }
  }

  std::cerr << "[getConcreteRLWESchemeParam] logqi after sanitization: [";
  for (size_t i = 0; i < logqi.size(); ++i)
    std::cerr << logqi[i] << (i + 1 < logqi.size() ? ", " : "");
  std::cerr << "]\n";

  auto maxLogqi = *std::max_element(logqi.begin(), logqi.end());
  std::vector<double> logpi(ceil(static_cast<double>(logqi.size()) / dnum),
                            maxLogqi);

  std::cerr << "[getConcreteRLWESchemeParam] maxLogqi=" << maxLogqi
            << ", logpi size=" << logpi.size()
            << ", logpi value=" << maxLogqi << "\n";

  double logPQ = std::accumulate(logqi.begin(), logqi.end(), 0.0) +
                 std::accumulate(logpi.begin(), logpi.end(), 0.0);

  std::cerr << "[getConcreteRLWESchemeParam] Initial logPQ=" << logPQ << "\n";

  auto ringDim = computeRingDim(logPQ, minRingDim);
  std::cerr << "[getConcreteRLWESchemeParam] Initial ringDim=" << ringDim << "\n";

  std::vector<int64_t> qiImpl;
  std::vector<int64_t> piImpl;
  bool redo = false;
  int iteration = 0;
  do {
    redo = false;
    qiImpl.clear();
    piImpl.clear();
    ++iteration;

    std::cerr << "[getConcreteRLWESchemeParam] --- Iteration " << iteration
              << " | ringDim=" << ringDim << " ---\n";

    std::vector<int64_t> existingPrimes;
    if (plaintextModulus != 0) {
      std::cerr << "[getConcreteRLWESchemeParam] Reserving plaintextModulus="
                << plaintextModulus << " as existing prime\n";
      existingPrimes.push_back(plaintextModulus);
    }

    double newLogPQ = 0;

    std::cerr << "[getConcreteRLWESchemeParam] Finding " << logqi.size()
              << " qi primes...\n";
    for (size_t i = 0; i < logqi.size(); ++i) {
      auto prime = findPrime(logqi[i], ringDim, existingPrimes);
      std::cerr << "[getConcreteRLWESchemeParam]   qi[" << i << "]: prime="
                << prime << " (log2=" << log2(prime) << ")\n";
      qiImpl.push_back(prime);
      existingPrimes.push_back(prime);
      newLogPQ += log2(prime);
    }

    std::cerr << "[getConcreteRLWESchemeParam] Finding " << logpi.size()
              << " pi primes...\n";
    for (size_t i = 0; i < logpi.size(); ++i) {
      auto prime = findPrime(logpi[i], ringDim, existingPrimes);
      std::cerr << "[getConcreteRLWESchemeParam]   pi[" << i << "]: prime="
                << prime << " (log2=" << log2(prime) << ")\n";
      piImpl.push_back(prime);
      existingPrimes.push_back(prime);
      newLogPQ += log2(prime);
    }

    std::cerr << "[getConcreteRLWESchemeParam] newLogPQ=" << newLogPQ << "\n";

    auto newRingDim = computeRingDim(newLogPQ, minRingDim);
    if (newRingDim != ringDim) {
      std::cerr << "[getConcreteRLWESchemeParam] ringDim too small: "
                << ringDim << " -> " << newRingDim << ", redoing...\n";
      ringDim = newRingDim;
      redo = true;
    } else {
      std::cerr << "[getConcreteRLWESchemeParam] ringDim=" << ringDim
                << " is sufficient, done.\n";
    }
  } while (redo);

  // update logqi and logpi
  logqi.clear();
  logpi.clear();
  for (auto qi : qiImpl) logqi.push_back(log2(qi));
  for (auto pi : piImpl) logpi.push_back(log2(pi));

  std::cerr << "[getConcreteRLWESchemeParam] Final | ringDim=" << ringDim
            << ", qi count=" << qiImpl.size()
            << ", pi count=" << piImpl.size() << "\n";

  return RLWESchemeParam(ringDim, level, logqi, qiImpl, dnum, logpi, piImpl,
                         usePublicKey, encryptionTechniqueExtended);
}

void RLWESchemeParam::print(llvm::raw_ostream& os) const {
  os << "ringDim: " << ringDim << "\n";
  os << "level: " << level << "\n";
  os << "logqi: ";
  for (auto qi : logqi) {
    os << doubleToString2Prec(qi) << " ";
  }
  os << "\n";
  os << "qi: ";
  for (auto qi : qi) {
    os << qi << " ";
  }
  os << "\n";
  os << "dnum: " << dnum << "\n";
  os << "logpi: ";
  for (auto pi : logpi) {
    os << doubleToString2Prec(pi) << " ";
  }
  os << "\n";
  os << "pi: ";
  for (auto pi : pi) {
    os << pi << " ";
  }
  os << "\n";
  os << "usePublicKey: " << usePublicKey << "\n";
  os << "encryptionTechniqueExtended: " << encryptionTechniqueExtended << "\n";
}

}  // namespace heir
}  // namespace mlir
