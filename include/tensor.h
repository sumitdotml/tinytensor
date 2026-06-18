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
int tt_tensor_fill(tt_tensor *tensor, float value);
int tt_tensor_set_flat(tt_tensor *tensor, size_t index, float value);
int tt_tensor_get_flat(tt_tensor *tensor, size_t index, float *out);

#endif
