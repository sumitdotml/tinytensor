#ifndef TINYTENSOR_TENSOR_H
#define TINYTENSOR_TENSOR_H

#include <stddef.h>

#define TT_TENSOR_MAX_DIMS 8

typedef struct tt_tensor {
  float *data;
  size_t ndim;
  size_t shape[TT_TENSOR_MAX_DIMS];
  size_t strides[TT_TENSOR_MAX_DIMS];
  size_t numel;
} tt_tensor;

int tt_tensor_create(tt_tensor *tensor, size_t ndim, const size_t *shape);
void tt_tensor_free(tt_tensor *tensor);

#endif
