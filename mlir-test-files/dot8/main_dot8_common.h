#ifndef DOT8_COMMON_H
#define DOT8_COMMON_H

#include <cstdint>
#include <iostream>
#include <vector>

#include "../params.h"
#include "src/pke/include/openfhe.h"  // from @openfhe

#define EvalNoiseBGV NoiseEvaluator::EvalNoiseBGV

// If printModulusChain is already defined in params.h, don't declare it again
// Otherwise, declare it here
// void printModulusChain(const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc);

// Common main function implementation that uses the specified variant's functions
template <typename FuncGenerator, typename FuncConfigure, typename FuncEncryptArg0,
          typename FuncEncryptArg1, typename FuncCompute, typename FuncDecryptResult>
int run(FuncGenerator generateCryptoContext,
                  FuncConfigure configureCryptoContext,
                  FuncEncryptArg0 encryptArg0,
                  FuncEncryptArg1 encryptArg1,
                  FuncCompute compute,
                  FuncDecryptResult decryptResult,
                  const std::string selectionApproach) {
  auto cc = generateCryptoContext();
  auto keyPair = cc->KeyGen();
  cc = configureCryptoContext(cc, keyPair.secretKey);

  std::cout << *(cc->GetCryptoParameters()) << std::endl;

  // Use the printModulusChain function from params.h
  printModulusChain(cc, "dot8", selectionApproach);
  saveParamsToJsonFile(cc, "dot8", selectionApproach);

  std::vector<int16_t> arg0 = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<int16_t> arg1 = {2, 3, 4, 5, 6, 7, 8, 9};

  // Alternative test data
  // std::vector<int16_t> arg0 = {100, 100, 100, 100, 100, 100, 100, 100};
  // std::vector<int16_t> arg1 = {100, 100, 100, 100, 100, 100, 100, 100};

  int64_t expected = 250;

  auto arg0Encrypted = encryptArg0(cc, arg0, keyPair.publicKey);
  auto arg1Encrypted = encryptArg1(cc, arg1, keyPair.publicKey);

  auto outputEncrypted = compute(cc, arg0Encrypted, arg1Encrypted);

  int16_t actual = decryptResult(cc, outputEncrypted, keyPair.secretKey);

  std::cout << "Expected: " << expected << "\n";
  std::cerr << "Actual: " << actual << "\n";

  return actual != expected;
}

#endif // DOT8_COMMON_H
