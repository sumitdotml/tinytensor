#include "tensor.h"
#include <stdio.h>

static int expect_size(const char *name, size_t actual, size_t expected) {
  if (actual != expected) {
    fprintf(stderr, "%s: expected %zu, got %zu\n", name, expected, actual);
    return 1;
  }

  return 0;
}

int main(void) {
  tt_tensor tensor = {0};
  const size_t shape[] = {2, 3, 4};

  if (tt_tensor_create(&tensor, 3, shape) != 0) {
    fprintf(stderr, "tt_tensor_create failed\n");
    return 1;
  }

  if (tt_tensor_fill(&tensor, 3.5f) != 0) {
    fprintf(stderr, "tt_tensor_fill failed\n");
    tt_tensor_free(&tensor);
    return 1;
  }

  for (size_t i = 0; i < tensor.numel; ++i) {
    if (tensor.data[i] != 3.5f) {
      fprintf(stderr, "data[%zu]: expected 3.500000, got %f\n", i, tensor.data[i]);
      tt_tensor_free(&tensor);
      return 1;
    }
  }

  printf("fill: all %zu flat values set to 3.500000\n", tensor.numel);

  if (tt_tensor_set_flat(&tensor, 5, 42.0f) != 0) {
    fprintf(stderr, "tt_tensor_set_flat failed\n");
    tt_tensor_free(&tensor);
    return 1;
  }

  float value = 0.0f;
  if (tt_tensor_get_flat(&tensor, 5, &value) != 0) {
    fprintf(stderr, "tt_tensor_get_flat failed\n");
    tt_tensor_free(&tensor);
    return 1;
  }

  if (value != 42.0f) {
    fprintf(stderr, "flat value: expected 42.000000, got %f\n", value);
    tt_tensor_free(&tensor);
    return 1;
  }

  printf("flat access: data[5] = %f\n", value);

  const size_t indices[] = {1, 2, 3};
  size_t offset = 0;
  if (tt_tensor_offset(&tensor, indices, 3, &offset) != 0) {
    fprintf(stderr, "tt_tensor_offset failed\n");
    tt_tensor_free(&tensor);
    return 1;
  }

  if (expect_size("offset", offset, 23) != 0) {
    tt_tensor_free(&tensor);
    return 1;
  }

  printf("offset: indices=[1, 2, 3] -> flat offset=%zu\n", offset);

  if (tt_tensor_set(&tensor, indices, 3, 99.0f) != 0) {
    fprintf(stderr, "tt_tensor_set failed\n");
    tt_tensor_free(&tensor);
    return 1;
  }

  float multidim_value = 0.0f;
  if (tt_tensor_get(&tensor, indices, 3, &multidim_value) != 0) {
    fprintf(stderr, "tt_tensor_get failed\n");
    tt_tensor_free(&tensor);
    return 1;
  }

  if (multidim_value != 99.0f) {
    fprintf(stderr, "tensor[1,2,3]: expected 99.000000, got %f\n", multidim_value);
    tt_tensor_free(&tensor);
    return 1;
  }

  printf("multidim access: tensor[1,2,3] = %f\n", multidim_value);

  tt_tensor_free(&tensor);
  printf("tensor access test passed\n");
  return 0;
}
