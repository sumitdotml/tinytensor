#include "tensor.h"
#include <stddef.h>
#include <stdio.h>

static void print_softmax_groups(const tt_tensor *tensor, const tt_tensor *out) {
  size_t last_dim_len = tensor->shape[tensor->ndim - 1];
  size_t groups = tensor->numel / last_dim_len;

  for (size_t g = 0; g < groups; ++g) {
    size_t offset = g * last_dim_len;
    float sum = 0.0f;

    printf("group %zu input:  [", g);
    for (size_t i = 0; i < last_dim_len; ++i) {
      printf("%s%.6f", i == 0 ? "" : ", ", tensor->data[offset + i]);
    }
    printf("]\n");

    printf("group %zu output: [", g);
    for (size_t i = 0; i < last_dim_len; ++i) {
      float value = out->data[offset + i];
      sum += value;
      printf("%s%.6f", i == 0 ? "" : ", ", value);
    }
    printf("] sum=%.6f\n\n", sum);
  }
}

int main(void) {
  tt_tensor tensor = {0};
  tt_tensor out = {0};
  const size_t shape_tensor[] = {2, 3, 4};
  const size_t shape_out[] = {2, 3, 4};
  const size_t ndim_tensor = 3;
  const size_t ndim_out = 3;

  if (tt_tensor_create(&tensor, ndim_tensor, shape_tensor) != 0 ||
      tt_tensor_create(&out, ndim_out, shape_out) != 0) {
    fprintf(stderr, "failed to create tensors for softmax demo\n");
    tt_tensor_free(&tensor);
    tt_tensor_free(&out);
    return 1;
  }

  const float tensor_values[] = {1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,
                                 9.0f,  10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f,
                                 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f};

  for (size_t i = 0; i < tensor.numel; ++i) {
    if (tt_tensor_set_flat(&tensor, i, tensor_values[i]) != 0) {
      fprintf(stderr, "failed to prepare tensor for softmax test\n");
      tt_tensor_free(&tensor);
      tt_tensor_free(&out);
      return 1;
    }
  }

  if (tt_tensor_softmax_last_dim(&tensor, &out) != 0) {
    fprintf(stderr, "tt_tensor_softmax_last_dim failed\n");
    tt_tensor_free(&tensor);
    tt_tensor_free(&out);
    return 1;
  }

  print_softmax_groups(&tensor, &out);

  tt_tensor_free(&tensor);
  tt_tensor_free(&out);
  return 0;
}
