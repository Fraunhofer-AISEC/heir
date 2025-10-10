#include <cstdint>
#include <vector>
#include <iostream>

#include "src/pke/include/openfhe.h"  // from @openfhe
#include "addi_i8.h"
#include "scheme-test/bench/bench.hpp"

int main(int argc, char *argv[]) {
    CryptoContext<DCRTPoly> cryptoContext = foo__generate_crypto_context();

    KeyPair<DCRTPoly> keyPair;
    keyPair = cryptoContext->KeyGen();

    cryptoContext = foo__configure_crypto_context(cryptoContext, keyPair.secretKey);

    int8_t arg0 = 1;
    int8_t arg1 = 2;
    int8_t expected = 1;

    auto arg0Encrypted = foo__encrypt__arg0(cryptoContext, arg0, keyPair.publicKey);
    auto arg1Encrypted = foo__encrypt__arg1(cryptoContext, arg1, keyPair.publicKey);

    // Correctness check
    auto outputEncrypted = foo(cryptoContext, arg0Encrypted, arg1Encrypted);
    auto actual = foo__decrypt__result0(cryptoContext, outputEncrypted, keyPair.secretKey);
    std::cout << "Expected: " << expected << "\n";
    std::cout << "Actual: " << actual << "\n";

    // Benchmark foo (excluding encryption/decryption)
    const size_t warmup = 5;
    const size_t iters  = 10;
    auto stats = bench::run(
        [&]() {
            return foo(cryptoContext, arg0Encrypted, arg1Encrypted);
        },
        warmup, iters
    );

    bench::print(stats, "foo");

    return 0;
}