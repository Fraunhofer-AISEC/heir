#include <cstdint>
#include <iostream>
#include <vector>

#include "../params.h"
#include "mult5.h"
#include "src/pke/include/openfhe.h"  // from @openfhe

#define EvalNoiseBGV NoiseEvaluator::EvalNoiseBGV

int main(int argc, char* argv[]) {
  auto cc = func__generate_crypto_context();
  auto keyPair = cc->KeyGen();
  cc = func__configure_crypto_context(cc, keyPair.secretKey);

  std::cout << *(cc->GetCryptoParameters()) << std::endl;

  printModulusChain(cc);

  std::vector<int32_t> arg0 = {1, 2, 3, 4,};
  std::vector<int32_t> arg1 = {2, 3, 4, 5,};

  std::vector<int32_t> expected = {4, 72, 432, 1600};

  auto arg0Encrypted = func__encrypt__arg0(cc, arg0, keyPair.publicKey);
  auto arg1Encrypted = func__encrypt__arg1(cc, arg1, keyPair.publicKey);

  auto outputEncrypted = func(cc, arg0Encrypted, arg1Encrypted);

  std::vector<int32_t> actual =
      func__decrypt__result0(cc, outputEncrypted, keyPair.secretKey);

  std::cout << "Expected: " << expected << "\n";
  std::cerr << "Actual: " << actual << "\n";

  return actual != expected;
}
