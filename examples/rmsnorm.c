#include "tensor.h"
#include <stddef.h>
#include <stdio.h>

static void print_rmsnorm_groups(const tt_tensor *tensor, const tt_tensor *gamma,
                                 const tt_tensor *out) {
  size_t last_dim_len = tensor->shape[tensor->ndim - 1];
  size_t groups = tensor->numel / last_dim_len;

  printf("gamma: [");
  for (size_t i = 0; i < gamma->numel; ++i) {
    printf("%s%.6f", i == 0 ? "" : ", ", gamma->data[i]);
  }
  printf("]\n\n");

  for (size_t g = 0; g < groups; ++g) {
    size_t offset = g * last_dim_len;

    printf("group %zu input:  [", g);
    for (size_t i = 0; i < last_dim_len; ++i) {
      printf("%s%.6f", i == 0 ? "" : ", ", tensor->data[offset + i]);
    }
    printf("]\n");

    printf("group %zu output: [", g);
    for (size_t i = 0; i < last_dim_len; ++i) {
      printf("%s%.6f", i == 0 ? "" : ", ", out->data[offset + i]);
    }
    printf("]\n\n");
  }
}

int main(void) {
  tt_tensor tensor = {0};
  tt_tensor gamma = {0};
  tt_tensor out = {0};
  const size_t tensor_shape[] = {2, 2};
  const size_t gamma_shape[] = {2};

  if (tt_tensor_create(&tensor, 2, tensor_shape) != 0 ||
      tt_tensor_create(&gamma, 1, gamma_shape) != 0 ||
      tt_tensor_create(&out, 2, tensor_shape) != 0) {
    fprintf(stderr, "failed to create tensors for RMSNorm demo\n");
    tt_tensor_free(&tensor);
    tt_tensor_free(&gamma);
    tt_tensor_free(&out);
    return 1;
  }

  if (tt_tensor_set_flat(&tensor, 0, 3.0f) != 0 || tt_tensor_set_flat(&tensor, 1, 4.0f) != 0 ||
      tt_tensor_set_flat(&tensor, 2, 6.0f) != 0 || tt_tensor_set_flat(&tensor, 3, 8.0f) != 0 ||
      tt_tensor_set_flat(&gamma, 0, 1.0f) != 0 || tt_tensor_set_flat(&gamma, 1, 2.0f) != 0) {
    fprintf(stderr, "failed to prepare tensors for RMSNorm demo\n");
    tt_tensor_free(&tensor);
    tt_tensor_free(&gamma);
    tt_tensor_free(&out);
    return 1;
  }

  if (tt_tensor_rmsnorm_last_dim(&tensor, &gamma, &out, 0.0f) != 0) {
    fprintf(stderr, "tt_tensor_rmsnorm_last_dim failed\n");
    tt_tensor_free(&tensor);
    tt_tensor_free(&gamma);
    tt_tensor_free(&out);
    return 1;
  }

  print_rmsnorm_groups(&tensor, &gamma, &out);

  tt_tensor_free(&tensor);
  tt_tensor_free(&gamma);
  tt_tensor_free(&out);
  return 0;
}
