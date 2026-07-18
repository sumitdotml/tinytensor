/*
 * Modal run:
 * make modal SRC=cuda_playground/cuda_add.cu
 * image: nvidia/cuda:12.8.1-devel-ubuntu24.04
 * GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition
 * compute capability: 12.0
 *
 * shape=[256, 256] numel=65536 runs=10 total=0.016575s avg=1657.533100us
 * first 8 outputs:
 * out[0] = 100.000000
 * out[1] = 102.000000
 * out[2] = 104.000000
 * out[3] = 106.000000
 * out[4] = 108.000000
 * out[5] = 110.000000
 * out[6] = 112.000000
 * out[7] = 114.000000
 * cuda add playground test passed
 */

#include "cuda_helpers.h"

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define TT_TENSOR_MAX_DIMS 8

typedef struct tt_tensor {
  float *data;
  size_t ndim;
  size_t shape[TT_TENSOR_MAX_DIMS];
  size_t numel;
} tt_tensor;

static double now_seconds(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}

static int tensor_create(tt_tensor *tensor, size_t rows, size_t cols) {
  tensor->ndim = 2;
  tensor->shape[0] = rows;
  tensor->shape[1] = cols;
  tensor->numel = rows * cols;
  tensor->data = (float *)malloc(tensor->numel * sizeof(float));
  return tensor->data == NULL;
}

static void tensor_free(tt_tensor *tensor) {
  free(tensor->data);
  tensor->data = NULL;
  tensor->numel = 0;
}

static int same_shape(const tt_tensor *a, const tt_tensor *b, const tt_tensor *out) {
  return a != NULL && b != NULL && out != NULL && a->data != NULL && b->data != NULL &&
         out->data != NULL && a->ndim == b->ndim && a->ndim == out->ndim &&
         a->numel == b->numel && a->numel == out->numel && a->shape[0] == b->shape[0] &&
         a->shape[0] == out->shape[0] && a->shape[1] == b->shape[1] &&
         a->shape[1] == out->shape[1];
}

__global__ void tensor_add_kernel(const float *a, const float *b, float *out, size_t numel) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < numel) {
    out[i] = a[i] + b[i];
  }
}

static int tensor_add_cuda(const tt_tensor *a, const tt_tensor *b, tt_tensor *out, size_t runs) {
  if (!same_shape(a, b, out)) {
    return 1;
  }

  float *a_d = NULL;
  float *b_d = NULL;
  float *out_d = NULL;
  size_t numel = a->numel;
  size_t bytes = numel * sizeof(float);
  const size_t threads_per_block = 256;
  size_t block_count = (numel + threads_per_block - 1) / threads_per_block;
  dim3 grid((unsigned int)block_count, 1, 1);
  dim3 block((unsigned int)threads_per_block, 1, 1);
  double start = 0.0;
  double elapsed = 0.0;
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

  start = now_seconds();
  for (size_t run = 0; run < runs; ++run) {
    tensor_add_kernel<<<grid, block>>>(a_d, b_d, out_d, numel);
    if (cudaGetLastError() != cudaSuccess) {
      goto cleanup;
    }
  }

  if (cudaMemcpy(out->data, out_d, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
    goto cleanup;
  }
  elapsed = now_seconds() - start;

  printf("shape=[%zu, %zu] numel=%zu runs=%zu total=%.6fs avg=%.6fus\n", a->shape[0],
         a->shape[1], numel, runs, elapsed, (elapsed / (double)runs) * 1000000.0);

  status = 0;

cleanup:
  cudaFree(a_d);
  cudaFree(b_d);
  cudaFree(out_d);
  return status;
}

int main(void) {
  const size_t rows = 256;
  const size_t cols = 256;
  const size_t runs = 10;

  tt_tensor a = {0};
  tt_tensor b = {0};
  tt_tensor out = {0};

  if (tt_cuda_print_device_info() != 0) {
    fprintf(stderr, "failed to read CUDA device info\n");
    return 1;
  }

  if (tensor_create(&a, rows, cols) != 0 || tensor_create(&b, rows, cols) != 0 ||
      tensor_create(&out, rows, cols) != 0) {
    fprintf(stderr, "failed to create tensors\n");
    tensor_free(&a);
    tensor_free(&b);
    tensor_free(&out);
    return 1;
  }

  for (size_t i = 0; i < a.numel; ++i) {
    a.data[i] = (float)(i % 17);
    b.data[i] = (float)(100 + (i % 23));
  }

  if (tensor_add_cuda(&a, &b, &out, runs) != 0) {
    fprintf(stderr, "tensor_add_cuda failed\n");
    tensor_free(&a);
    tensor_free(&b);
    tensor_free(&out);
    return 1;
  }

  for (size_t i = 0; i < out.numel; ++i) {
    float expected = a.data[i] + b.data[i];
    if (out.data[i] != expected) {
      fprintf(stderr, "out[%zu]: expected %f, got %f\n", i, expected, out.data[i]);
      tensor_free(&a);
      tensor_free(&b);
      tensor_free(&out);
      return 1;
    }
  }

  printf("first 8 outputs:\n");
  for (size_t i = 0; i < 8; ++i) {
    printf("out[%zu] = %f\n", i, out.data[i]);
  }
  printf("cuda add playground test passed\n");

  tensor_free(&a);
  tensor_free(&b);
  tensor_free(&out);
  return 0;
}
