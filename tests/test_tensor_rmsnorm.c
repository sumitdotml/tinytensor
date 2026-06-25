#include "tensor.h"
#include <stdio.h>

static int close_float(float a, float b) {
  float diff = a - b;
  if (diff < 0.0f) {
    diff = -diff;
  }
  return diff < 0.00001f;
}

int main(void) {
  tt_tensor input = {0};
  tt_tensor gamma = {0};
  tt_tensor out = {0};
  const size_t input_shape[] = {2, 2};
  const size_t gamma_shape[] = {2};
  int status = 1;

  if (tt_tensor_create(&input, 2, input_shape) != 0 ||
      tt_tensor_create(&gamma, 1, gamma_shape) != 0 ||
      tt_tensor_create(&out, 2, input_shape) != 0) {
    fprintf(stderr, "failed to create tensors for RMSNorm test\n");
    goto cleanup;
  }

  if (tt_tensor_set_flat(&input, 0, 3.0f) != 0 || tt_tensor_set_flat(&input, 1, 4.0f) != 0 ||
      tt_tensor_set_flat(&input, 2, 6.0f) != 0 || tt_tensor_set_flat(&input, 3, 8.0f) != 0 ||
      tt_tensor_set_flat(&gamma, 0, 1.0f) != 0 || tt_tensor_set_flat(&gamma, 1, 2.0f) != 0) {
    fprintf(stderr, "failed to prepare tensors for RMSNorm test\n");
    goto cleanup;
  }

  if (tt_tensor_rmsnorm_last_dim(&input, &gamma, &out, 0.0f) != 0) {
    fprintf(stderr, "tt_tensor_rmsnorm_last_dim failed\n");
    goto cleanup;
  }

  if (!close_float(out.data[0], 0.848528f) || !close_float(out.data[1], 2.262742f) ||
      !close_float(out.data[2], 0.848528f) || !close_float(out.data[3], 2.262742f)) {
    fprintf(stderr, "RMSNorm output mismatch: [%f, %f, %f, %f]\n", out.data[0], out.data[1],
            out.data[2], out.data[3]);
    goto cleanup;
  }

  printf("rmsnorm: [2, 2] tensor with gamma [1, 2] -> expected normalized values\n");
  status = 0;

cleanup:
  tt_tensor_free(&input);
  tt_tensor_free(&gamma);
  tt_tensor_free(&out);
  return status;
}
