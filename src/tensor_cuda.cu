#include "tensor_cuda.h"
#include <cuda_runtime.h>

static __global__ void tensor_add_kernel(const float *a, const float *b, float *out,
                                         size_t numel) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < numel) {
    out[i] = a[i] + b[i];
  }
}

static int check_same_shape_cuda_op(const tt_tensor *a, const tt_tensor *b,
                                    const tt_tensor *out) {
  if (a == nullptr || b == nullptr || out == nullptr || a->data == nullptr ||
      b->data == nullptr || out->data == nullptr || a->numel != b->numel ||
      a->numel != out->numel || a->ndim != b->ndim || a->ndim != out->ndim) {
    return 1;
  }

  for (size_t dim = 0; dim < out->ndim; ++dim) {
    if (a->shape[dim] != b->shape[dim] || a->shape[dim] != out->shape[dim]) {
      return 1;
    }
  }

  return 0;
}

int tt_tensor_add_cuda(const tt_tensor *a, const tt_tensor *b, tt_tensor *out) {
  if (check_same_shape_cuda_op(a, b, out) != 0) {
    return 1;
  }

  float *a_d = nullptr;
  float *b_d = nullptr;
  float *out_d = nullptr;
  size_t numel = a->numel;
  size_t bytes = numel * sizeof(float);
  const size_t threads_per_block = 256;
  size_t block_count = (numel + threads_per_block - 1) / threads_per_block;
  dim3 grid((unsigned int)block_count, 1, 1);
  dim3 block((unsigned int)threads_per_block, 1, 1);
  int status = 1;

  if (cudaMalloc((void **)&a_d, bytes) != cudaSuccess ||
      cudaMalloc((void **)&b_d, bytes) != cudaSuccess ||
      cudaMalloc((void **)&out_d, bytes) != cudaSuccess) {
    goto cleanup;
  }

  if (cudaMemcpy(a_d, a->data, bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
      cudaMemcpy(b_d, b->data, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
    goto cleanup;
  }

  tensor_add_kernel<<<grid, block>>>(a_d, b_d, out_d, numel);
  if (cudaGetLastError() != cudaSuccess) {
    goto cleanup;
  }

  if (cudaMemcpy(out->data, out_d, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
    goto cleanup;
  }

  status = 0;

cleanup:
  cudaFree(a_d);
  cudaFree(b_d);
  cudaFree(out_d);
  return status;
}
