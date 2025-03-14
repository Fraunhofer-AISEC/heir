#include <cstdint>  // Include for int64_t
#include <fstream>
#include <iostream>
#include <vector>

#define DIM 1024
#define FRACTIONAL_BITS 16  // Number of bits for the fractional part
#define SCALE_FACTOR (1 << FRACTIONAL_BITS)  // 2^16 as int64_t

template <typename T, size_t N>
struct MemRefDescriptor {
  T *allocated;
  T *aligned;
  intptr_t offset;
  intptr_t sizes[N];
  intptr_t strides[N];
};

extern "C" {
void *_mlir_ciface_mlp(MemRefDescriptor<int64_t, 2> *output,
                       MemRefDescriptor<int64_t, 2> *input,
                       MemRefDescriptor<int64_t, 2> *fc1,
                       MemRefDescriptor<int64_t, 2> *fc2,
                       MemRefDescriptor<int64_t, 2> *fc1_buffer,
                       MemRefDescriptor<int64_t, 2> *fc2_buffer);
}

struct Sample {
  int label;
  float image[DIM];
};

using Dataset = std::vector<Sample>;

void load_dataset(Dataset &dataset, const char *filename) {
  std::ifstream file(filename);
  Sample sample;
  while (file >> sample.label) {
    for (int i = 0; i < DIM; i++) {
      file >> sample.image[i];
    }
    dataset.push_back(sample);
  }
}

void load_weight(const char *filename, int64_t w1[DIM * DIM],
                 int64_t w2[DIM * DIM]) {
  std::ifstream file(filename);
  for (int i = 0; i < DIM; i++) {
    for (int j = 0; j < DIM; j++) {
      float weight;
      file >> weight;
      w1[i * DIM + j] = static_cast<int64_t>(weight * SCALE_FACTOR);
    }
  }
  for (int i = 0; i < DIM; i++) {
    for (int j = 0; j < DIM; j++) {
      float weight;
      file >> weight;
      w2[i * DIM + j] = static_cast<int64_t>(weight * SCALE_FACTOR);
    }
  }
}

template <int N>
int argmax(int64_t *A) {
  int max_idx = 0;
  for (int i = 1; i < N; i++) {
    if (A[i] > A[max_idx]) {
      max_idx = i;
    }
  }
  return max_idx;
}

int main(int argc, char *argv[]) {
  auto *fc1 = new int64_t[DIM * DIM];
  auto *fc2 = new int64_t[DIM * DIM];

  auto *fc1_buffer = new int64_t[DIM];
  auto *fc2_buffer = new int64_t[DIM];

  load_weight("relu_net_reg.txt", fc1, fc2);

  auto dataset = Dataset();
  load_dataset(dataset, "mnist_test.txt");

  int accurate = 0;
  int total = 10000;

  for (int i = 0; i < total; ++i) {
    auto *input = dataset[i].image;

    for (int i = 0; i < DIM; i++) {
      fc1_buffer[i] = 0;
      fc2_buffer[i] = 0;
    }

    // Encode input floats to fixed-point int64_t
    int64_t input_fixed[DIM];
    for (int j = 0; j < DIM; j++) {
      input_fixed[j] = static_cast<int64_t>(input[j] * SCALE_FACTOR);
    }

    MemRefDescriptor<int64_t, 2> input_ref = {
        input_fixed,  // allocated
        input_fixed,  // aligned
        0,            // offset
        {1, DIM},     // sizes[N]
        {DIM, 1},     // strides[N]
    };
    MemRefDescriptor<int64_t, 2> fc1_ref = {
        fc1,         // allocated
        fc1,         // aligned
        0,           // offset
        {DIM, DIM},  // sizes[N]
        {DIM, 1},    // strides[N]
    };
    MemRefDescriptor<int64_t, 2> fc2_ref = {
        fc2,         // allocated
        fc2,         // aligned
        0,           // offset
        {DIM, DIM},  // sizes[N]
        {DIM, 1},    // strides[N]
    };
    MemRefDescriptor<int64_t, 2> fc1_buffer_ref = {
        fc1_buffer,  // allocated
        fc1_buffer,  // aligned
        0,           // offset
        {1, DIM},    // sizes[N]
        {DIM, 1},    // strides[N]
    };
    MemRefDescriptor<int64_t, 2> fc2_buffer_ref = {
        fc2_buffer,  // allocated
        fc2_buffer,  // aligned
        0,           // offset
        {1, DIM},    // sizes[N]
        {DIM, 1},    // strides[N]
    };

    MemRefDescriptor<int64_t, 2> output_ref;

    _mlir_ciface_mlp(&output_ref, &input_ref, &fc1_ref, &fc2_ref,
                     &fc1_buffer_ref, &fc2_buffer_ref);

    auto *output = output_ref.allocated;

    // Decode output back to float
    float output_float[DIM];
    for (int j = 0; j < DIM; j++) {
      output_float[j] = static_cast<float>(output[j]) / SCALE_FACTOR;
    }

    auto max_id = argmax<DIM>(output);
    auto label = dataset[i].label;

    if (max_id == label) {
      accurate++;
    }
    if (i % 100 == 0) {
      std::cout << "accuracy: " << accurate << "/" << i << std::endl;
    }
  }

  std::cout << "accuracy: " << accurate << "/" << total << std::endl;

  delete[] fc1;
  delete[] fc2;
  delete[] fc1_buffer;
  delete[] fc2_buffer;

  return 0;
}
