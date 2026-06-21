#include "tensor.h"
#include <stdio.h>

int main(void) {
  tt_tensor a = {0};
  tt_tensor b = {0};
  tt_tensor result = {0};
  const size_t a_shape[] = {2, 3};
  const size_t b_shape[] = {3, 2};
  const size_t result_shape[] = {2, 2};

  if (tt_tensor_create(&a, 2, a_shape) != 0 || tt_tensor_create(&b, 2, b_shape) != 0 ||
      tt_tensor_create(&result, 2, result_shape) != 0) {
    fprintf(stderr, "failed to create tensors for matmul test\n");
    tt_tensor_free(&a);
    tt_tensor_free(&b);
    tt_tensor_free(&result);
    return 1;
  }

  const float a_values[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  const float b_values[] = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
  const float expected[] = {58.0f, 64.0f, 139.0f, 154.0f};

  for (size_t i = 0; i < a.numel; ++i) {
    if (tt_tensor_set_flat(&a, i, a_values[i]) != 0) {
      fprintf(stderr, "failed to prepare left tensor for matmul test\n");
      tt_tensor_free(&a);
      tt_tensor_free(&b);
      tt_tensor_free(&result);
      return 1;
    }
  }

  for (size_t i = 0; i < b.numel; ++i) {
    if (tt_tensor_set_flat(&b, i, b_values[i]) != 0) {
      fprintf(stderr, "failed to prepare right tensor for matmul test\n");
      tt_tensor_free(&a);
      tt_tensor_free(&b);
      tt_tensor_free(&result);
      return 1;
    }
  }

  if (tt_tensor_matmul(&a, &b, &result) != 0) {
    fprintf(stderr, "tt_tensor_matmul failed\n");
    tt_tensor_free(&a);
    tt_tensor_free(&b);
    tt_tensor_free(&result);
    return 1;
  }

  for (size_t i = 0; i < result.numel; ++i) {
    if (result.data[i] != expected[i]) {
      fprintf(stderr, "result.data[%zu]: expected %f, got %f\n", i, expected[i],
              result.data[i]);
      tt_tensor_free(&a);
      tt_tensor_free(&b);
      tt_tensor_free(&result);
      return 1;
    }
  }

  printf("matmul: [2, 3] @ [3, 2] -> [58, 64, 139, 154]\n");

  tt_tensor_free(&a);
  tt_tensor_free(&b);
  tt_tensor_free(&result);

  printf("tensor matmul test passed\n");
  return 0;
}
