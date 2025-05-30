#ifndef ADD_DEPTH_0_COMMON_H
#define ADD_DEPTH_0_COMMON_H

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
            const std::string selectionApproach,
            bool ignoreComputation = false) {
  auto cc = generateCryptoContext();
  auto keyPair = cc->KeyGen();
  cc = configureCryptoContext(cc, keyPair.secretKey);

  std::cout << *(cc->GetCryptoParameters()) << std::endl;

  printModulusChain(cc, "add-eq-10", selectionApproach);
  saveParamsToJsonFile(cc, "add-eq-10", selectionApproach);

  if (ignoreComputation) {
    return 0;
  }

  // Initialize 64 tensors with values
  std::vector<std::vector<int16_t>> args;

  // Create 64 tensor arguments
  for (int i = 0; i < 64; i++) {
    // Initialize each tensor with 8 elements
    // Using i+1 as the value for all elements in tensor i
    std::vector<int16_t> arg(8, i % 64 == 0 ? 1 : 0);
    args.push_back(arg);
  }

  // Calculate expected result
  int64_t sum_per_position = 0;
  for (int i = 0; i < 64; i++) {
    auto pos = i % 64 == 0 ? 1 : 0;

    sum_per_position += pos;  // Each tensor has value i at all positions
  }

  // The expected result is a vector where each element is sum_per_position^2
  std::vector<int16_t> expected_vector(8, sum_per_position);

  // Encrypt all arguments
  std::vector<ConstCiphertext<DCRTPoly>> encryptedArgs;
  for (int i = 0; i < 64; i++) {
    // Fix the encryption function call - use the numbered version corresponding
    // to each arg
    auto encrypted = encryptArg0(cc, args[i], keyPair.publicKey);
    encryptedArgs.push_back(encrypted);
  }

  // Call the function with all encrypted arguments
  auto outputEncrypted = compute(
      cc, encryptedArgs[0], encryptedArgs[1], encryptedArgs[2],
      encryptedArgs[3], encryptedArgs[4], encryptedArgs[5], encryptedArgs[6],
      encryptedArgs[7], encryptedArgs[8], encryptedArgs[9], encryptedArgs[10],
      encryptedArgs[11], encryptedArgs[12], encryptedArgs[13],
      encryptedArgs[14], encryptedArgs[15], encryptedArgs[16],
      encryptedArgs[17], encryptedArgs[18], encryptedArgs[19],
      encryptedArgs[20], encryptedArgs[21], encryptedArgs[22],
      encryptedArgs[23], encryptedArgs[24], encryptedArgs[25],
      encryptedArgs[26], encryptedArgs[27], encryptedArgs[28],
      encryptedArgs[29], encryptedArgs[30], encryptedArgs[31],
      encryptedArgs[32], encryptedArgs[33], encryptedArgs[34],
      encryptedArgs[35], encryptedArgs[36], encryptedArgs[37],
      encryptedArgs[38], encryptedArgs[39], encryptedArgs[40],
      encryptedArgs[41], encryptedArgs[42], encryptedArgs[43],
      encryptedArgs[44], encryptedArgs[45], encryptedArgs[46],
      encryptedArgs[47], encryptedArgs[48], encryptedArgs[49],
      encryptedArgs[50], encryptedArgs[51], encryptedArgs[52],
      encryptedArgs[53], encryptedArgs[54], encryptedArgs[55],
      encryptedArgs[56], encryptedArgs[57], encryptedArgs[58],
      encryptedArgs[59], encryptedArgs[60], encryptedArgs[61],
      encryptedArgs[62], encryptedArgs[63]);

  // Decrypt the result - this will be a vector of 8 values
  auto actual_vector = decryptResult(cc, outputEncrypted, keyPair.secretKey);

  // Print and compare results
  std::cout << "Expected: [";
  for (size_t i = 0; i < expected_vector.size(); i++) {
    std::cout << expected_vector[i];
    if (i < expected_vector.size() - 1) std::cout << ", ";
  }
  std::cout << "]\n";

  std::cerr << "Actual: [";
  for (size_t i = 0; i < actual_vector.size(); i++) {
    std::cerr << actual_vector[i];
    if (i < actual_vector.size() - 1) std::cerr << ", ";
  }
  std::cerr << "]\n";

  // Check if all elements match
  bool all_match = true;
  for (size_t i = 0; i < expected_vector.size(); i++) {
    if (actual_vector[i] != expected_vector[i]) {
      all_match = false;
      break;
    }
  }

  return !all_match;
}

#endif // ADD_DEPTH_0_COMMON_H
