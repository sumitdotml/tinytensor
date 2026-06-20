#include "tensor.h"
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
  for (size_t i = 0; i < nindices; ++i) { // nindices here = tensor->ndim
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

int tt_tensor_add(const tt_tensor *a, const tt_tensor *b, tt_tensor *out) {
  if (a == NULL || a->data == NULL || b == NULL || b->data == NULL || a->numel != b->numel ||
      a->ndim != b->ndim || out == NULL || out->data == NULL || a->numel != out->numel ||
      a->ndim != out->ndim) {
    return 1;
  }

  for (size_t dim = 0; dim < out->ndim; ++dim) {
    if (a->shape[dim] != b->shape[dim] || a->shape[dim] != out->shape[dim]) {
      return 1;
    }
  }
  for (size_t ele = 0; ele < out->numel; ++ele) {
    out->data[ele] = a->data[ele] + b->data[ele];
  }
  return 0;
}

int tt_tensor_mul(const tt_tensor *a, const tt_tensor *b, tt_tensor *out) {
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
  for (size_t ele = 0; ele < out->numel; ++ele) {
    out->data[ele] = a->data[ele] * b->data[ele];
  }
  return 0;
}

// sum over all the elements within the tensor
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
