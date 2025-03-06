#include <cstdint>
#include <iostream>
#include <vector>

#include "add-64-4_bisection.h"
#include "main_add-64-4_common.h"

int main(int argc, char* argv[]) {
    return run(
      func__generate_crypto_context,
      func__configure_crypto_context,
      func__encrypt__arg0,
      func__encrypt__arg1,
      func,
      func__decrypt__result0,
      "bisection"
  );
}
