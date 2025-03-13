#include <cstdint>
#include <vector>

#include "gmock/gmock.h"  // from @googletest
#include "gtest/gtest.h"  // from @googletest

// Generated headers (block clang-format from messing up order)
#include "tests/Examples/openfhe/bgv/roberts_cross/roberts_cross_64x64_lib.h"

using ::testing::ContainerEq;

namespace mlir {
namespace heir {
namespace openfhe {

TEST(RobertsCrossTest, TestInput1) {
  auto cryptoContext = roberts_cross__generate_crypto_context();
  auto keyPair = cryptoContext->KeyGen();
  auto publicKey = keyPair.publicKey;
  auto secretKey = keyPair.secretKey;
  cryptoContext =
      roberts_cross__configure_crypto_context(cryptoContext, secretKey);

  std::vector<int16_t> input;
  std::vector<int16_t> expected;
  input.reserve(4096);
  expected.reserve(4096);

  for (int i = 0; i < 4096; ++i) {
    input.push_back(i);
  }

  for (int row = 0; row < 64; ++row) {
    for (int col = 0; col < 64; ++col) {
      // (img[x-1][y-1] - img[x][y])^2 + (img[x-1][y] - img[x][y-1])^2
      int xY = (row * 64 + col) % 4096;
      int xYm1 = (row * 64 + col - 1) % 4096;
      int xm1Y = ((row - 1) * 64 + col) % 4096;
      int xm1Ym1 = ((row - 1) * 64 + col - 1) % 4096;

      if (xYm1 < 0) xYm1 += 4096;
      if (xm1Y < 0) xm1Y += 4096;
      if (xm1Ym1 < 0) xm1Ym1 += 4096;

      int16_t v1 = (input[xm1Ym1] - input[xY]);
      int16_t v2 = (input[xm1Y] - input[xYm1]);
      int16_t sum = v1 * v1 + v2 * v2;
      expected.push_back(sum);
    }
  }

  auto inputEncrypted =
      roberts_cross__encrypt__arg0(cryptoContext, input, keyPair.publicKey);
  auto outputEncrypted = roberts_cross(cryptoContext, inputEncrypted);
  auto actual = roberts_cross__decrypt__result0(cryptoContext, outputEncrypted,
                                                keyPair.secretKey);

  EXPECT_THAT(actual, ContainerEq(expected));
}

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir
