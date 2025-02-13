#include <cstdint>
#include <iostream>
#include <vector>

#include "../params.h"
#include "psi.h"
#include "src/pke/include/openfhe.h"  // from @openfhe

#define EvalNoiseBGV NoiseEvaluator::EvalNoiseBGV

int main(int argc, char* argv[]) {
  auto cc = func__generate_crypto_context();
  auto keyPair = cc->KeyGen();
  cc = func__configure_crypto_context(cc, keyPair.secretKey);

  std::cout << *(cc->GetCryptoParameters()) << std::endl;

  printModulusChain(cc);

  std::vector<int32_t> arg0 = {1, 2, 3, 4,}; // 5, 6, 7, 8};
  std::vector<int32_t> arg1 = {3, 4, 4, 5,}; // 6, 7, 8, 9};
  std::vector<int32_t> arg2 = {76, 4, 1, 19,}; // 7, 8, 190, 19};

  // std::vector<int16_t> arg0 = {100, 100, 100, 100, 100, 100, 100, 100};
  // std::vector<int16_t> arg1 = {100, 100, 100, 100, 100, 100, 100, 100};

  std::vector<int32_t> expected = {0, 0, 0, 8}; // 4, 5, 6, 7, 8, 9};

  auto arg0Encrypted = func__encrypt__arg0(cc, arg0, keyPair.publicKey);
  auto arg1Encrypted = func__encrypt__arg1(cc, arg1, keyPair.publicKey);
  auto arg2Encrypted = func__encrypt__arg2(cc, arg2, keyPair.publicKey);

  auto outputEncrypted = func(cc, arg0Encrypted, arg1Encrypted, arg2Encrypted);

  std::vector<int32_t> actual =
      func__decrypt__result0(cc, outputEncrypted, keyPair.secretKey);

  std::cout << "Expected: " << expected << "\n";
  std::cerr << "Actual: " << actual << "\n";

  return actual != expected;
}
