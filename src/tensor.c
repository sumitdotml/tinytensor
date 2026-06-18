#include "tensor.h"
#include <stdint.h>
#include <stdlib.h>

static void tt_tensor_reset(tt_tensor *tensor) {
  tensor->data = NULL;
  tensor->ndim = 0;
  tensor->numel = 0;

  for (size_t i = 0; i < TT_TENSOR_MAX_DIMS; ++i) {
    tensor->shape[i] = 0;
    tensor->strides[i] = 0;
  }
}

static int checked_mul_size(size_t a, size_t b, size_t *out) {
  if (a != 0 && b > SIZE_MAX / a) {
    return 1;
  }

  *out = a * b;
  return 0;
}

int tt_tensor_create(tt_tensor *tensor, size_t ndim, const size_t *shape) {
  size_t numel = 1;

  tt_tensor_reset(tensor);

  if (ndim == 0 || ndim > TT_TENSOR_MAX_DIMS || shape == NULL) {
    return 1;
  }

  for (size_t i = 0; i < ndim; ++i) {
    if (shape[i] == 0 || checked_mul_size(numel, shape[i], &numel) != 0) {
      return 1;
    }
    tensor->shape[i] = shape[i];
  }

  tensor->strides[ndim - 1] = 1;
  for (size_t i = ndim - 1; i > 0; --i) {
    tensor->strides[i - 1] = tensor->strides[i] * tensor->shape[i];
  }

  tensor->data = malloc(numel * sizeof *tensor->data);
  if (tensor->data == NULL) {
    tt_tensor_reset(tensor);
    return 1;
  }

  tensor->ndim = ndim;
  tensor->numel = numel;
  return 0;
}

void tt_tensor_free(tt_tensor *tensor) {
  free(tensor->data);
  tt_tensor_reset(tensor);
}
