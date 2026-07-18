#include "cuda_helpers.h"
#include "tensor.h"
#include "tensor_cuda.h"
#include <stdio.h>

int main(void) {
  tt_tensor a = {0};
  tt_tensor b = {0};
  tt_tensor result = {0};
  const size_t shape[] = {2, 3};

  if (tt_cuda_print_device_info() != 0) {
    fprintf(stderr, "failed to read CUDA device info\n");
    return 1;
  }

  if (tt_tensor_create(&a, 2, shape) != 0 || tt_tensor_create(&b, 2, shape) != 0 ||
      tt_tensor_create(&result, 2, shape) != 0) {
    fprintf(stderr, "failed to create tensors for CUDA add test\n");
    tt_tensor_free(&a);
    tt_tensor_free(&b);
    tt_tensor_free(&result);
    return 1;
  }

  for (size_t i = 0; i < a.numel; ++i) {
    if (tt_tensor_set_flat(&a, i, (float)i) != 0 ||
        tt_tensor_set_flat(&b, i, (float)(10 + i)) != 0) {
      fprintf(stderr, "failed to prepare tensors for CUDA add test\n");
      tt_tensor_free(&a);
      tt_tensor_free(&b);
      tt_tensor_free(&result);
      return 1;
    }
  }

  if (tt_tensor_add_cuda(&a, &b, &result) != 0) {
    fprintf(stderr, "tt_tensor_add_cuda failed\n");
    tt_tensor_free(&a);
    tt_tensor_free(&b);
    tt_tensor_free(&result);
    return 1;
  }

  for (size_t i = 0; i < result.numel; ++i) {
    float expected = (float)i + (float)(10 + i);
    if (result.data[i] != expected) {
      fprintf(stderr, "result.data[%zu]: expected %f, got %f\n", i, expected,
              result.data[i]);
      tt_tensor_free(&a);
      tt_tensor_free(&b);
      tt_tensor_free(&result);
      return 1;
    }
  }

  printf("cuda add: [2, 3] tensors -> expected elementwise sums\n");

  tt_tensor_free(&a);
  tt_tensor_free(&b);
  tt_tensor_free(&result);

  printf("tensor CUDA add test passed\n");
  return 0;
}
