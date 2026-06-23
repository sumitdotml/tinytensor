#include "tensor.h"
#include <math.h>
#include <stddef.h>
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

// for now, just a tensor filler without strides or shape math
int tt_tensor_fill(tt_tensor *tensor, float value) {
  if (tensor == NULL || tensor->data == NULL) {
    return 1;
  }
  for (size_t ele = 0; ele < tensor->numel; ++ele) {
    tensor->data[ele] = value;
  }
  return 0;
}

// naive flat element access
int tt_tensor_set_flat(tt_tensor *tensor, size_t index, float value) {
  if (tensor == NULL || tensor->data == NULL || index >= tensor->numel) {
    return 1;
  }
  tensor->data[index] = value;
  return 0;
}

int tt_tensor_get_flat(tt_tensor *tensor, size_t index, float *out) {
  if (tensor == NULL || tensor->data == NULL || index >= tensor->numel || out == NULL) {
    return 1;
  }
  *out = tensor->data[index];
  return 0;
}

int tt_tensor_offset(const tt_tensor *tensor, const size_t *indices, size_t nindices, size_t *out) {
  if (tensor == NULL || tensor->data == NULL || nindices != tensor->ndim || out == NULL ||
      indices == NULL) {
    return 1;
  }
  *out = 0;
  for (size_t i = 0; i < nindices; ++i) {
    if (!(indices[i] < tensor->shape[i])) {
      return 1;
    }
    *out += indices[i] * tensor->strides[i];
  }
  return 0;
}

// multidimensional tensor accessor: set
int tt_tensor_set(tt_tensor *tensor, const size_t *indices, size_t nindices, float value) {
  if (tensor == NULL || tensor->data == NULL || nindices != tensor->ndim) {
    return 1;
  }

  size_t offset;
  if (tt_tensor_offset(tensor, indices, nindices, &offset) != 0) {
    return 1;
  }
  tensor->data[offset] = value;
  return 0;
}

// multidimensional tensor accessor: get
int tt_tensor_get(const tt_tensor *tensor, const size_t *indices, size_t nindices, float *out) {
  if (tensor == NULL || tensor->data == NULL || nindices != tensor->ndim || out == NULL) {
    return 1;
  }
  size_t offset;
  if (tt_tensor_offset(tensor, indices, nindices, &offset) != 0) {
    return 1;
  }
  *out = tensor->data[offset];
  return 0;
}

static int check_same_shape_binary_op(const tt_tensor *a, const tt_tensor *b,
                                      const tt_tensor *out) {
  if (a == NULL || b == NULL || out == NULL || a->data == NULL || b->data == NULL ||
      out->data == NULL || a->numel != b->numel || a->numel != out->numel || a->ndim != b->ndim ||
      a->ndim != out->ndim) {
    return 1;
  }

  for (size_t dim = 0; dim < out->ndim; ++dim) {
    if (a->shape[dim] != b->shape[dim] || a->shape[dim] != out->shape[dim]) {
      return 1;
    }
  }

  return 0;
}

int tt_tensor_add(const tt_tensor *a, const tt_tensor *b, tt_tensor *out) {
  if (check_same_shape_binary_op(a, b, out) != 0) {
    return 1;
  }

  for (size_t ele = 0; ele < out->numel; ++ele) {
    out->data[ele] = a->data[ele] + b->data[ele];
  }
  return 0;
}

int tt_tensor_mul(const tt_tensor *a, const tt_tensor *b, tt_tensor *out) {
  if (check_same_shape_binary_op(a, b, out) != 0) {
    return 1;
  }

  for (size_t ele = 0; ele < out->numel; ++ele) {
    out->data[ele] = a->data[ele] * b->data[ele];
  }
  return 0;
}

int tt_tensor_sum(const tt_tensor *tensor, float *out) {
  if (tensor == NULL || tensor->data == NULL || out == NULL) {
    return 1;
  }
  float sum = 0.0f;
  for (size_t ele = 0; ele < tensor->numel; ++ele) {
    sum += tensor->data[ele];
  }
  *out = sum;
  return 0;
}

int tt_tensor_max(const tt_tensor *tensor, float *out) {
  if (tensor == NULL || tensor->data == NULL || out == NULL) {
    return 1;
  }
  float element = tensor->data[0];
  for (size_t i = 1; i < tensor->numel; ++i) {
    float current = tensor->data[i];
    if (current >= element) {
      element = current;
    }
  }
  *out = element;
  return 0;
}

int tt_tensor_mean(const tt_tensor *tensor, float *out) {
  if (tensor == NULL || tensor->data == NULL || out == NULL) {
    return 1;
  }
  float sum = 0.0f;
  if (tt_tensor_sum(tensor, &sum) != 0) {
    return 1;
  }
  *out = sum / tensor->numel;
  return 0;
}

// Matmul for 2D tensors
int tt_tensor_matmul(const tt_tensor *a, const tt_tensor *b, tt_tensor *out) {
  if (a == NULL || b == NULL || out == NULL || a->data == NULL || b->data == NULL ||
      out->data == NULL || a->ndim != 2 || b->ndim != 2 || out->ndim != 2) {
    return 1;
  }
  if (a->shape[1] != b->shape[0] || a->shape[0] != out->shape[0] || b->shape[1] != out->shape[1]) {
    return 1;
  }
  for (size_t row_a = 0; row_a < a->shape[0]; ++row_a) {
    for (size_t col_b = 0; col_b < b->shape[1]; ++col_b) {
      float sum = 0.0f;
      for (size_t k = 0; k < a->shape[1]; ++k) {
        size_t a_offset = row_a * a->strides[0] + k * a->strides[1];
        size_t b_offset = k * b->strides[0] + col_b * b->strides[1];
        sum += a->data[a_offset] * b->data[b_offset];
      }
      size_t out_offset = row_a * out->strides[0] + col_b * out->strides[1];
      out->data[out_offset] = sum;
    }
  }
  return 0;
}

int tt_tensor_softmax_last_dim(const tt_tensor *tensor, tt_tensor *out) {
  if (tensor == NULL || out == NULL || tensor->ndim == 0 || out->ndim == 0 ||
      tensor->data == NULL || out->data == NULL || tensor->ndim != out->ndim ||
      tensor->numel != out->numel) {
    return 1;
  }
  for (size_t dim = 0; dim < tensor->ndim; ++dim) {
    if (tensor->shape[dim] != out->shape[dim]) {
      return 1;
    }
  }

  size_t last_dim_len = tensor->shape[tensor->ndim - 1];

  /*
    Here, groups = total groups with each containing `last_dim_len` elements
    so for instance, if our tensor is:

    [[[0.8474, 0.5893, 0.5541, 0.3149],
    [0.4477, 0.3493, 0.8349, 0.4417],
    [0.4945, 0.7338, 0.7073, 0.5568]],

    [[0.4150, 0.4761, 0.1175, 0.6317],
    [0.7603, 0.9477, 0.0183, 0.2032],
    [0.7804, 0.1790, 0.7569, 0.1477]]]

    That means its shape is [2, 3, 4]. Which means:
    - ndim = 3
    - last_dim_len = tensor[-1] = tensor[ndim - 1] = 4
    - numel = 2 * 3 * 4 = 24
    - groups = numel / last_dim_len = 24 / 4 = 6

    So, each group here would be something like:
    - group 0 = [0.8474, 0.5893, 0.5541, 0.3149]
    - group 1 = [0.4477, 0.3493, 0.8349, 0.4417]
    ... and so on.
  */
  size_t groups = tensor->numel / last_dim_len;

  for (size_t g = 0; g < groups; ++g) {
    size_t offset = g * last_dim_len;

    // max calculation per group
    float max = tensor->data[offset];
    for (size_t i = 1; i < last_dim_len; ++i) {
      float ele = tensor->data[offset + i];
      if (ele > max) {
        max = ele;
      }
    }

    // exp sum per group
    float exp_sum = 0;
    for (size_t i = 0; i < last_dim_len; ++i) {
      float ele = tensor->data[offset + i];
      exp_sum += expf(ele - max);
    }

    // normalize inside each group
    for (size_t i = 0; i < last_dim_len; ++i) {
      float ele = tensor->data[offset + i];
      // for a row, each softmaxed element at index i = exp(value_at_i - max_value) / exp_sum
      out->data[offset + i] = expf(ele - max) / exp_sum;
    }
  }
  return 0;
}
