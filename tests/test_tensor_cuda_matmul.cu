#include "tensor.h"
#include "tensor_cuda.h"
#include <stdio.h>

int main(void) {
  tt_tensor a = {};
  tt_tensor b = {};
  tt_tensor result = {};
  const size_t a_shape[] = {2, 3};
  const size_t b_shape[] = {3, 4};
  const size_t result_shape[] = {2, 4};
  const float expected[] = {38.0f, 44.0f, 50.0f, 56.0f,
                            83.0f, 98.0f, 113.0f, 128.0f};
  int status = 1;

  if (tt_tensor_create(&a, 2, a_shape) != 0 ||
      tt_tensor_create(&b, 2, b_shape) != 0 ||
      tt_tensor_create(&result, 2, result_shape) != 0) {
    fprintf(stderr, "failed to create tensors for CUDA matmul test\n");
    goto cleanup;
  }

  // A contains 1..6; B contains 1..12, both in row-major order.
  for (size_t i = 0; i < a.numel; ++i) {
    a.data[i] = (float)(i + 1);
  }
  for (size_t i = 0; i < b.numel; ++i) {
    b.data[i] = (float)(i + 1);
  }

  if (tt_tensor_matmul_cuda(&a, &b, &result) != 0) {
    fprintf(stderr, "tt_tensor_matmul_cuda failed\n");
    goto cleanup;
  }

  for (size_t i = 0; i < result.numel; ++i) {
    if (result.data[i] != expected[i]) {
      fprintf(stderr, "result.data[%zu]: expected %f, got %f\n",
              i, expected[i], result.data[i]);
      goto cleanup;
    }
  }

  // [2, 3] @ [2, 4] has incompatible inner dimensions.
  if (tt_tensor_matmul_cuda(&a, &result, &result) == 0) {
    fprintf(stderr, "CUDA matmul accepted incompatible inner dimensions\n");
    goto cleanup;
  }

  // A's [2, 3] shape is not the required [2, 4] output shape.
  if (tt_tensor_matmul_cuda(&a, &b, &a) == 0) {
    fprintf(stderr, "CUDA matmul accepted an incorrect output shape\n");
    goto cleanup;
  }

  printf("cuda matmul: [2, 3] @ [3, 4] ->\n");
  printf("38 44 50 56\n83 98 113 128\n");
  printf("invalid inner dimensions and output shape rejected\n");
  printf("tensor CUDA matmul test passed\n");
  status = 0;

cleanup:
  tt_tensor_free(&a);
  tt_tensor_free(&b);
  tt_tensor_free(&result);
  return status;
}
