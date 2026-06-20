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

  /* checking tensor creation and metadata */
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

  printf("created tensor: shape=[2, 3, 4], strides=[12, 4, 1], numel=24\n");

  /* checking fill */
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

  /* checking flat access */
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

  /* checking multidimensional offset */
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

  /* checking multidimensional access */
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

  /* checking elementwise add */
  tt_tensor a = {0};
  tt_tensor b = {0};
  tt_tensor result = {0};
  const size_t add_shape[] = {2, 3};

  if (tt_tensor_create(&a, 2, add_shape) != 0 || tt_tensor_create(&b, 2, add_shape) != 0 ||
      tt_tensor_create(&result, 2, add_shape) != 0) {
    fprintf(stderr, "failed to create tensors for add test\n");
    tt_tensor_free(&a);
    tt_tensor_free(&b);
    tt_tensor_free(&result);
    tt_tensor_free(&tensor);
    return 1;
  }

  if (tt_tensor_fill(&a, 1.5f) != 0 || tt_tensor_fill(&b, 2.0f) != 0 ||
      tt_tensor_add(&a, &b, &result) != 0) {
    fprintf(stderr, "tt_tensor_add failed\n");
    tt_tensor_free(&a);
    tt_tensor_free(&b);
    tt_tensor_free(&result);
    tt_tensor_free(&tensor);
    return 1;
  }

  for (size_t i = 0; i < result.numel; ++i) {
    if (result.data[i] != 3.5f) {
      fprintf(stderr, "result.data[%zu]: expected 3.500000, got %f\n", i, result.data[i]);
      tt_tensor_free(&a);
      tt_tensor_free(&b);
      tt_tensor_free(&result);
      tt_tensor_free(&tensor);
      return 1;
    }
  }

  printf("add: [2, 3] tensors 1.500000 + 2.000000 -> 3.500000\n");

  /* checking elementwise multiply */
  if (tt_tensor_fill(&a, 3.0f) != 0 || tt_tensor_fill(&b, 2.0f) != 0 ||
      tt_tensor_mul(&a, &b, &result) != 0) {
    fprintf(stderr, "tt_tensor_mul failed\n");
    tt_tensor_free(&a);
    tt_tensor_free(&b);
    tt_tensor_free(&result);
    tt_tensor_free(&tensor);
    return 1;
  }

  for (size_t i = 0; i < result.numel; ++i) {
    if (result.data[i] != 6.0f) {
      fprintf(stderr, "result.data[%zu]: expected 6.000000, got %f\n", i, result.data[i]);
      tt_tensor_free(&a);
      tt_tensor_free(&b);
      tt_tensor_free(&result);
      tt_tensor_free(&tensor);
      return 1;
    }
  }

  printf("multiply: [2, 3] tensors 3.000000 * 2.000000 -> 6.000000\n");

  /* checking full tensor sum */
  if (tt_tensor_fill(&a, 2.0f) != 0) {
    fprintf(stderr, "tt_tensor_fill failed for sum test\n");
    tt_tensor_free(&a);
    tt_tensor_free(&b);
    tt_tensor_free(&result);
    tt_tensor_free(&tensor);
    return 1;
  }

  float total = 0.0f;
  if (tt_tensor_sum(&a, &total) != 0) {
    fprintf(stderr, "tt_tensor_sum failed\n");
    tt_tensor_free(&a);
    tt_tensor_free(&b);
    tt_tensor_free(&result);
    tt_tensor_free(&tensor);
    return 1;
  }

  if (total != 12.0f) {
    fprintf(stderr, "sum: expected 12.000000, got %f\n", total);
    tt_tensor_free(&a);
    tt_tensor_free(&b);
    tt_tensor_free(&result);
    tt_tensor_free(&tensor);
    return 1;
  }

  printf("sum: [2, 3] tensor filled with 2.000000 -> 12.000000\n");

  /* checking full tensor max */
  if (tt_tensor_fill(&a, -5.0f) != 0 || tt_tensor_set_flat(&a, 4, -2.0f) != 0) {
    fprintf(stderr, "failed to prepare tensor for max test\n");
    tt_tensor_free(&a);
    tt_tensor_free(&b);
    tt_tensor_free(&result);
    tt_tensor_free(&tensor);
    return 1;
  }

  float max_value = 0.0f;
  if (tt_tensor_max(&a, &max_value) != 0) {
    fprintf(stderr, "tt_tensor_max failed\n");
    tt_tensor_free(&a);
    tt_tensor_free(&b);
    tt_tensor_free(&result);
    tt_tensor_free(&tensor);
    return 1;
  }

  if (max_value != -2.0f) {
    fprintf(stderr, "max: expected -2.000000, got %f\n", max_value);
    tt_tensor_free(&a);
    tt_tensor_free(&b);
    tt_tensor_free(&result);
    tt_tensor_free(&tensor);
    return 1;
  }

  printf("max: [2, 3] tensor with values -5.000000 and -2.000000 -> -2.000000\n");

  /* checking full tensor mean */
  for (size_t i = 0; i < a.numel; ++i) {
    if (tt_tensor_set_flat(&a, i, (float)(i + 1)) != 0) {
      fprintf(stderr, "failed to prepare tensor for mean test\n");
      tt_tensor_free(&a);
      tt_tensor_free(&b);
      tt_tensor_free(&result);
      tt_tensor_free(&tensor);
      return 1;
    }
  }

  float mean = 0.0f;
  if (tt_tensor_mean(&a, &mean) != 0) {
    fprintf(stderr, "tt_tensor_mean failed\n");
    tt_tensor_free(&a);
    tt_tensor_free(&b);
    tt_tensor_free(&result);
    tt_tensor_free(&tensor);
    return 1;
  }

  if (mean != 3.5f) {
    fprintf(stderr, "mean: expected 3.500000, got %f\n", mean);
    tt_tensor_free(&a);
    tt_tensor_free(&b);
    tt_tensor_free(&result);
    tt_tensor_free(&tensor);
    return 1;
  }

  printf("mean: [2, 3] tensor with values 1..6 -> 3.500000\n");

  tt_tensor_free(&a);
  tt_tensor_free(&b);
  tt_tensor_free(&result);

  /* cleanup */
  tt_tensor_free(&tensor);

  if (tensor.data != NULL || tensor.ndim != 0 || tensor.numel != 0) {
    fprintf(stderr, "tt_tensor_free did not reset tensor state\n");
    return 1;
  }

  printf("tensor lifecycle test passed\n");
  return 0;
}
