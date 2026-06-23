#include "tensor.h"
#include <stdio.h>

/*
  Basic ops: elementwise add/mul and full-tensor reductions.
  Shape-specific ops such as matmul and softmax live in their own test files.
*/
int main(void) {
  tt_tensor a = {0};
  tt_tensor b = {0};
  tt_tensor result = {0};
  const size_t shape[] = {2, 3};

  if (tt_tensor_create(&a, 2, shape) != 0 || tt_tensor_create(&b, 2, shape) != 0 ||
      tt_tensor_create(&result, 2, shape) != 0) {
    fprintf(stderr, "failed to create tensors for basic ops test\n");
    tt_tensor_free(&a);
    tt_tensor_free(&b);
    tt_tensor_free(&result);
    return 1;
  }

  if (tt_tensor_fill(&a, 1.5f) != 0 || tt_tensor_fill(&b, 2.0f) != 0 ||
      tt_tensor_add(&a, &b, &result) != 0) {
    fprintf(stderr, "tt_tensor_add failed\n");
    tt_tensor_free(&a);
    tt_tensor_free(&b);
    tt_tensor_free(&result);
    return 1;
  }

  for (size_t i = 0; i < result.numel; ++i) {
    if (result.data[i] != 3.5f) {
      fprintf(stderr, "result.data[%zu]: expected 3.500000, got %f\n", i, result.data[i]);
      tt_tensor_free(&a);
      tt_tensor_free(&b);
      tt_tensor_free(&result);
      return 1;
    }
  }

  printf("add: [2, 3] tensors 1.500000 + 2.000000 -> 3.500000\n");

  if (tt_tensor_fill(&a, 3.0f) != 0 || tt_tensor_fill(&b, 2.0f) != 0 ||
      tt_tensor_mul(&a, &b, &result) != 0) {
    fprintf(stderr, "tt_tensor_mul failed\n");
    tt_tensor_free(&a);
    tt_tensor_free(&b);
    tt_tensor_free(&result);
    return 1;
  }

  for (size_t i = 0; i < result.numel; ++i) {
    if (result.data[i] != 6.0f) {
      fprintf(stderr, "result.data[%zu]: expected 6.000000, got %f\n", i, result.data[i]);
      tt_tensor_free(&a);
      tt_tensor_free(&b);
      tt_tensor_free(&result);
      return 1;
    }
  }

  printf("multiply: [2, 3] tensors 3.000000 * 2.000000 -> 6.000000\n");

  if (tt_tensor_fill(&a, 2.0f) != 0) {
    fprintf(stderr, "tt_tensor_fill failed for sum test\n");
    tt_tensor_free(&a);
    tt_tensor_free(&b);
    tt_tensor_free(&result);
    return 1;
  }

  float total = 0.0f;
  if (tt_tensor_sum(&a, &total) != 0) {
    fprintf(stderr, "tt_tensor_sum failed\n");
    tt_tensor_free(&a);
    tt_tensor_free(&b);
    tt_tensor_free(&result);
    return 1;
  }

  if (total != 12.0f) {
    fprintf(stderr, "sum: expected 12.000000, got %f\n", total);
    tt_tensor_free(&a);
    tt_tensor_free(&b);
    tt_tensor_free(&result);
    return 1;
  }

  printf("sum: [2, 3] tensor filled with 2.000000 -> 12.000000\n");

  if (tt_tensor_fill(&a, -5.0f) != 0 || tt_tensor_set_flat(&a, 4, -2.0f) != 0) {
    fprintf(stderr, "failed to prepare tensor for max test\n");
    tt_tensor_free(&a);
    tt_tensor_free(&b);
    tt_tensor_free(&result);
    return 1;
  }

  float max_value = 0.0f;
  if (tt_tensor_max(&a, &max_value) != 0) {
    fprintf(stderr, "tt_tensor_max failed\n");
    tt_tensor_free(&a);
    tt_tensor_free(&b);
    tt_tensor_free(&result);
    return 1;
  }

  if (max_value != -2.0f) {
    fprintf(stderr, "max: expected -2.000000, got %f\n", max_value);
    tt_tensor_free(&a);
    tt_tensor_free(&b);
    tt_tensor_free(&result);
    return 1;
  }

  printf("max: [2, 3] tensor with values -5.000000 and -2.000000 -> -2.000000\n");

  for (size_t i = 0; i < a.numel; ++i) {
    if (tt_tensor_set_flat(&a, i, (float)(i + 1)) != 0) {
      fprintf(stderr, "failed to prepare tensor for mean test\n");
      tt_tensor_free(&a);
      tt_tensor_free(&b);
      tt_tensor_free(&result);
      return 1;
    }
  }

  float mean = 0.0f;
  if (tt_tensor_mean(&a, &mean) != 0) {
    fprintf(stderr, "tt_tensor_mean failed\n");
    tt_tensor_free(&a);
    tt_tensor_free(&b);
    tt_tensor_free(&result);
    return 1;
  }

  if (mean != 3.5f) {
    fprintf(stderr, "mean: expected 3.500000, got %f\n", mean);
    tt_tensor_free(&a);
    tt_tensor_free(&b);
    tt_tensor_free(&result);
    return 1;
  }

  printf("mean: [2, 3] tensor with values 1..6 -> 3.500000\n");

  tt_tensor_free(&a);
  tt_tensor_free(&b);
  tt_tensor_free(&result);

  printf("tensor basic ops test passed\n");
  return 0;
}
