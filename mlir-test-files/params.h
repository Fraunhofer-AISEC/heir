#ifndef PARAMS_H
#define PARAMS_H

#include "src/pke/include/openfhe.h"  // from @openfhe

using namespace lbcrypto;

void printModulusChain(const CryptoContext<DCRTPoly>& cc) {
  auto modulusChain = cc->GetCryptoParameters()->GetElementParams()->GetParams();
  
  std::cout << "Modulus chain sizes (in bits): [";
  for (size_t i = 0; i < modulusChain.size(); ++i) {
    std::cout << modulusChain[i]->GetModulus().GetMSB();
    if (i != modulusChain.size() - 1) {
      std::cout << ", ";
    }
  }
  std::cout << "]" << std::endl;

  // Print the total size of the modulus chain
  std::cout << cc->GetCryptoParameters()->GetElementParams()->GetModulus().GetMSB() << std::endl;
  
}

#endif // PARAMS_H