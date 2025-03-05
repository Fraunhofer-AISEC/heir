#include <cstdint>
#include <iostream>
#include <vector>

#include "dot8_openfhe.h"
#include "main_dot8_common.h"

int main(int argc, char* argv[]) {
  return run_dot8_main(
      func__generate_crypto_context,
      func__configure_crypto_context,
      func__encrypt__arg0,
      func__encrypt__arg1,
      func,
      func__decrypt__result0,
      "openfhe"
  );
}
