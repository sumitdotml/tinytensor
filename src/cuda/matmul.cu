#include "tensor_cuda.h"
#include <cuda_runtime.h>

static __global__ void tensor_naive_matmul_kernel(const float *a_d, const float *b_d, float *out_d,
                                                  size_t num_rows, size_t num_inner, size_t num_cols) {
  size_t row_index = blockDim.y * blockIdx.y + threadIdx.y;
  size_t col_index = blockDim.x * blockIdx.x + threadIdx.x;

  float accumulator = 0.0f;
  if (row_index < num_rows && col_index < num_cols) {
    for (size_t k = 0; k < num_inner; ++k) {
      accumulator += a_d[row_index * num_inner + k] * b_d[k * num_cols + col_index];
    }
    out_d[row_index * num_cols + col_index] = accumulator;
  }
}

int tt_tensor_matmul_cuda(const tt_tensor *a, const tt_tensor *b, tt_tensor *out) {
  if (a == nullptr || b == nullptr || out == nullptr || a->data == nullptr || b->data == nullptr ||
      out->data == nullptr || a->shape[1] != b->shape[0] || a->shape[0] != out->shape[0] ||
      b->shape[1] != out->shape[1] || a->ndim != 2 || b->ndim != 2 || out->ndim != 2) {
    return 1;
  }
  size_t num_rows = a->shape[0];
  size_t num_cols = b->shape[1];
  size_t num_inner = a->shape[1];
  size_t threads_per_width = 3;  // across the "grid" of out
  size_t threads_per_height = 3; // down the "grid" of out

  size_t block_count_x = (num_cols + threads_per_width - 1) / threads_per_width;
  size_t block_count_y = (num_rows + threads_per_height - 1) / threads_per_height;
  size_t a_bytes = num_rows * num_inner * sizeof(float);
  size_t b_bytes = num_inner * num_cols * sizeof(float);
  size_t out_bytes = num_rows * num_cols * sizeof(float);

  float *a_d = nullptr, *b_d = nullptr, *out_d = nullptr;
  dim3 gridDim(block_count_x, block_count_y, 1);
  dim3 blockDim(threads_per_width, threads_per_height, 1);
  int status = 1;

  if (cudaMalloc((void **)&a_d, a_bytes) != cudaSuccess ||
      cudaMalloc((void **)&b_d, b_bytes) != cudaSuccess ||
      cudaMalloc((void **)&out_d, out_bytes) != cudaSuccess) {
    goto cleanup;
  }

  if (cudaMemcpy(a_d, a->data, a_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
      cudaMemcpy(b_d, b->data, b_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
    goto cleanup;
  }

  tensor_naive_matmul_kernel<<<gridDim, blockDim>>>(a_d, b_d, out_d, num_rows, num_inner, num_cols);
  if (cudaGetLastError() != cudaSuccess) {
    goto cleanup;
  }

  if (cudaMemcpy(out->data, out_d, out_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
    goto cleanup;
  }

  status = 0;

cleanup:
  cudaFree(a_d);
  cudaFree(b_d);
  cudaFree(out_d);
  return status;
}
