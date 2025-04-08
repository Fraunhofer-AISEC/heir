#include <cstdint>
#include <iostream>
#include <vector>

#include "add-eq-2_closed.h"
#include "main_add-eq-2_common.h"

int main(int argc, char* argv[]) {
  // Check for ignoreComputation flag
  bool ignoreComputation = false;
  if (argc > 1) {
      ignoreComputation = std::string(argv[1]) == "ignoreComputation";
  }
  return run(
    func__generate_crypto_context,
    func__configure_crypto_context,
    func__encrypt__arg0,
    func__encrypt__arg1,
    func,
    func__decrypt__result0,
    "closed",
    ignoreComputation
);
}

