#ifndef TINYTENSOR_TENSOR_CUDA_H
#define TINYTENSOR_TENSOR_CUDA_H

#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

int tt_tensor_add_cuda(const tt_tensor *a, const tt_tensor *b, tt_tensor *out);
int tt_tensor_matmul_cuda(const tt_tensor *a, const tt_tensor *b, tt_tensor *out);

#ifdef __cplusplus
}
#endif

#endif
