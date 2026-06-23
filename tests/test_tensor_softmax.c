#include "tensor.h"
#include <stdio.h>

static int close_float(float a, float b) {
  float diff = a - b;
  if (diff < 0.0f) {
    diff = -diff;
  }
  return diff < 0.00001f;
}

static int run_softmax_case(const size_t *shape, size_t ndim, const char *label) {
  tt_tensor input = {0};
  tt_tensor out = {0};
  int status = 1;

  if (tt_tensor_create(&input, ndim, shape) != 0 || tt_tensor_create(&out, ndim, shape) != 0) {
    fprintf(stderr, "failed to create tensors for %s softmax test\n", label);
    goto cleanup;
  }

  size_t last_dim = shape[ndim - 1];
  for (size_t i = 0; i < input.numel; ++i) {
    if (tt_tensor_set_flat(&input, i, (float)((i % last_dim) + 1)) != 0) {
      fprintf(stderr, "failed to prepare tensor for %s softmax test\n", label);
      goto cleanup;
    }
  }

  if (tt_tensor_softmax_last_dim(&input, &out) != 0) {
    fprintf(stderr, "tt_tensor_softmax_last_dim failed for %s tensor\n", label);
    goto cleanup;
  }

  size_t groups = input.numel / last_dim;
  for (size_t group = 0; group < groups; ++group) {
    size_t offset = group * last_dim;
    float sum = 0.0f;

    for (size_t i = 0; i < last_dim; ++i) {
      sum += out.data[offset + i];
      if (i > 0 && !(out.data[offset + i - 1] < out.data[offset + i])) {
        fprintf(stderr, "softmax %s group %zu is not increasing\n", label, group);
        goto cleanup;
      }
    }

    if (!close_float(sum, 1.0f)) {
      fprintf(stderr, "softmax %s group %zu sum: expected 1.000000, got %f\n", label, group, sum);
      goto cleanup;
    }
  }

  printf("softmax: %s tensor -> each last-dim group sums to 1.000000\n", label);
  status = 0;

cleanup:
  tt_tensor_free(&input);
  tt_tensor_free(&out);
  return status;
}

int main(void) {
  const size_t shape_1d[] = {3};
  const size_t shape_2d[] = {2, 3};
  const size_t shape_3d[] = {2, 2, 3};

  if (run_softmax_case(shape_1d, 1, "[3]") != 0 || run_softmax_case(shape_2d, 2, "[2, 3]") != 0 ||
      run_softmax_case(shape_3d, 3, "[2, 2, 3]") != 0) {
    return 1;
  }

  printf("tensor softmax test passed\n");
  return 0;
}
