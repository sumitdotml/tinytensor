/*
 * Modal run:
 * make modal SRC=cuda_playground/cuda_matmul.cu
 * image: nvidia/cuda:12.8.1-devel-ubuntu24.04
 * GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition
 * compute capability: 12.0
 *
 * cuda matmul: [2, 3] @ [3, 4] ->
 * 38.000000 44.000000 50.000000 56.000000
 * 83.000000 98.000000 113.000000 128.000000
 * invalid inner dimensions and output shape rejected
 * cuda matmul playground test passed
 */

#include "cuda_helpers.h"
#include <cuda_runtime.h>
#include <stdio.h>

#define TT_TENSOR_MAX_DIMS 8

typedef struct tt_tensor {
  float *data;
  size_t ndim;
  size_t shape[TT_TENSOR_MAX_DIMS];
  size_t strides[TT_TENSOR_MAX_DIMS];
  size_t numel;
} tt_tensor;

static __global__ void tensor_naive_matmul_kernel(const float *a_d, const float *b_d, float *out_d,
                                                  size_t num_rows, size_t num_inner,
                                                  size_t num_cols) {
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

static int tt_tensor_matmul_cuda(const tt_tensor *a, const tt_tensor *b, tt_tensor *out) {
  if (a == nullptr || b == nullptr || out == nullptr || a->data == nullptr || b->data == nullptr ||
      out->data == nullptr || a->shape[1] != b->shape[0] || a->shape[0] != out->shape[0] ||
      b->shape[1] != out->shape[1] || a->ndim != 2 || b->ndim != 2 || out->ndim != 2) {
    return 1;
  }
  size_t num_rows = a->shape[0];
  size_t num_cols = b->shape[1];
  size_t num_inner = a->shape[1];
  size_t threads_per_width = 3;
  size_t threads_per_height = 3;

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

int main(void) {
  float a_values[] = {1, 2, 3, 4, 5, 6};
  float b_values[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  float result_values[8] = {};
  const float expected[] = {38, 44, 50, 56, 83, 98, 113, 128};
  tt_tensor a = {a_values, 2, {2, 3}, {3, 1}, 6};
  tt_tensor b = {b_values, 2, {3, 4}, {4, 1}, 12};
  tt_tensor result = {result_values, 2, {2, 4}, {4, 1}, 8};

  if (tt_cuda_print_device_info() != 0) {
    fprintf(stderr, "failed to read CUDA device info\n");
    return 1;
  }

  if (tt_tensor_matmul_cuda(&a, &b, &result) != 0) {
    fprintf(stderr, "tt_tensor_matmul_cuda failed\n");
    return 1;
  }

  for (size_t i = 0; i < result.numel; ++i) {
    if (result.data[i] != expected[i]) {
      fprintf(stderr, "result.data[%zu]: expected %f, got %f\n", i, expected[i], result.data[i]);
      return 1;
    }
  }

  if (tt_tensor_matmul_cuda(&a, &result, &result) == 0) {
    fprintf(stderr, "CUDA matmul accepted incompatible inner dimensions\n");
    return 1;
  }
  if (tt_tensor_matmul_cuda(&a, &b, &a) == 0) {
    fprintf(stderr, "CUDA matmul accepted an incorrect output shape\n");
    return 1;
  }

  printf("cuda matmul: [2, 3] @ [3, 4] ->\n");
  for (size_t row = 0; row < result.shape[0]; ++row) {
    for (size_t col = 0; col < result.shape[1]; ++col) {
      printf("%f%s", result.data[row * result.shape[1] + col],
             col + 1 == result.shape[1] ? "\n" : " ");
    }
  }
  printf("invalid inner dimensions and output shape rejected\n");
  printf("cuda matmul playground test passed\n");
  return 0;
}
