#include <stdint.h>
#include <stdio.h>

struct Memref1D {
  int64_t *allocated;
  int64_t *aligned;
  int64_t offset;
  int64_t size;
  int64_t stride;
};

// This is the function we want to call from LLVM
// For mod-arith, the width is 64 bits
struct Memref1D roberts_cross(
    /* arg0*/
    int64_t *allocated, int64_t *aligned, int64_t offset, int64_t size,
    int64_t stride);

void memrefCopy();

int main() {
  int64_t input[4096];
  int64_t expected[4096];

  for (int i = 0; i < 4096; ++i) {
    input[i] = i;
  }

  for (int row = 0; row < 64; ++row) {
    for (int col = 0; col < 64; ++col) {
      // (img[x-1][y-1] - img[x][y])^2 + (img[x-1][y] - img[x][y-1])^2
      int64_t xY = (row * 64 + col) % 4096;
      int64_t xYm1 = (row * 64 + col - 1) % 4096;
      int64_t xm1Y = ((row - 1) * 64 + col) % 4096;
      int64_t xm1Ym1 = ((row - 1) * 64 + col - 1) % 4096;

      if (xYm1 < 0) xYm1 += 4096;
      if (xm1Y < 0) xm1Y += 4096;
      if (xm1Ym1 < 0) xm1Ym1 += 4096;

      int64_t v1 = (input[xm1Ym1] - input[xY]);
      int64_t v2 = (input[xm1Y] - input[xYm1]);
      int64_t sum = v1 * v1 + v2 * v2;
      expected[row * 64 + col] = sum;
    }
  }

  struct Memref1D memref = roberts_cross(
      /* arg 0*/
      input, input, 0, 4096, 1);

  int64_t *res = memref.aligned;

  for (int i = 0; i != 4096; ++i) {
    if (res[i] != expected[i]) {
      printf("Test failed at %d: %ld != %ld\n", i, res[i], expected[i]);
      return 0;
    }
  }
  printf("Test passed\n");
}
