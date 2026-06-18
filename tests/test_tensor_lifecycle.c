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

  if (tensor.data == NULL) {
    fprintf(stderr, "tensor data was not allocated\n");
    tt_tensor_free(&tensor);
    return 1;
  }

  if (expect_size("ndim", tensor.ndim, 3) != 0 ||
      expect_size("shape[0]", tensor.shape[0], 2) != 0 ||
      expect_size("shape[1]", tensor.shape[1], 3) != 0 ||
      expect_size("shape[2]", tensor.shape[2], 4) != 0 ||
      expect_size("strides[0]", tensor.strides[0], 12) != 0 ||
      expect_size("strides[1]", tensor.strides[1], 4) != 0 ||
      expect_size("strides[2]", tensor.strides[2], 1) != 0 ||
      expect_size("numel", tensor.numel, 24) != 0) {
    tt_tensor_free(&tensor);
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

  tt_tensor_free(&tensor);

  if (tensor.data != NULL || tensor.ndim != 0 || tensor.numel != 0) {
    fprintf(stderr, "tt_tensor_free did not reset tensor state\n");
    return 1;
  }

  printf("tensor lifecycle test passed\n");
  return 0;
}
