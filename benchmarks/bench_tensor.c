#include "tensor.h"
#include <stdio.h>
#include <time.h>

// making benchmark output visible enough that optimization cannot erase it freely
static volatile double bench_sink;

// measuring elapsed time with a monotonic clock
static double now_seconds(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}

// summing output values as a cheap benchmark sanity check
static double checksum(const tt_tensor *tensor) {
  double sum = 0.0;
  for (size_t i = 0; i < tensor->numel; ++i) {
    sum += tensor->data[i];
  }
  return sum;
}

// filling tensors with deterministic non-uniform values
static void fill_repeating(tt_tensor *tensor, float scale) {
  for (size_t i = 0; i < tensor->numel; ++i) {
    tensor->data[i] = (float)((i % 17) + 1) * scale;
  }
}

static int bench_matmul(void) {
  const size_t a_shape[] = {256, 256};
  const size_t b_shape[] = {256, 256};
  const size_t out_shape[] = {256, 256};
  const size_t runs = 5;
  tt_tensor a = {0};
  tt_tensor b = {0};
  tt_tensor out = {0};
  int status = 1;

  if (tt_tensor_create(&a, 2, a_shape) != 0 || tt_tensor_create(&b, 2, b_shape) != 0 ||
      tt_tensor_create(&out, 2, out_shape) != 0) {
    fprintf(stderr, "failed to create tensors for matmul benchmark\n");
    goto cleanup;
  }

  fill_repeating(&a, 0.01f);
  fill_repeating(&b, 0.02f);

  // warming up once before timed runs
  if (tt_tensor_matmul(&a, &b, &out) != 0) {
    fprintf(stderr, "matmul warmup failed\n");
    goto cleanup;
  }

  double start = now_seconds();
  for (size_t run = 0; run < runs; ++run) {
    if (tt_tensor_matmul(&a, &b, &out) != 0) {
      fprintf(stderr, "matmul benchmark failed\n");
      goto cleanup;
    }
  }
  double elapsed = now_seconds() - start;
  double sum = checksum(&out);
  bench_sink = sum;

  printf("matmul  [256, 256] @ [256, 256]  runs=%zu  total=%.6fs  avg=%.6fs  checksum=%.6f\n",
         runs, elapsed, elapsed / runs, sum);
  status = 0;

cleanup:
  tt_tensor_free(&a);
  tt_tensor_free(&b);
  tt_tensor_free(&out);
  return status;
}

static int bench_softmax(void) {
  const size_t shape[] = {512, 1024};
  const size_t runs = 25;
  tt_tensor input = {0};
  tt_tensor out = {0};
  int status = 1;

  if (tt_tensor_create(&input, 2, shape) != 0 || tt_tensor_create(&out, 2, shape) != 0) {
    fprintf(stderr, "failed to create tensors for softmax benchmark\n");
    goto cleanup;
  }

  fill_repeating(&input, 0.1f);

  // warming up once before timed runs
  if (tt_tensor_softmax_last_dim(&input, &out) != 0) {
    fprintf(stderr, "softmax warmup failed\n");
    goto cleanup;
  }

  double start = now_seconds();
  for (size_t run = 0; run < runs; ++run) {
    if (tt_tensor_softmax_last_dim(&input, &out) != 0) {
      fprintf(stderr, "softmax benchmark failed\n");
      goto cleanup;
    }
  }
  double elapsed = now_seconds() - start;
  double sum = checksum(&out);
  bench_sink = sum;

  printf("softmax [512, 1024] last dim       runs=%zu total=%.6fs  avg=%.6fs  checksum=%.6f\n",
         runs, elapsed, elapsed / runs, sum);
  status = 0;

cleanup:
  tt_tensor_free(&input);
  tt_tensor_free(&out);
  return status;
}

static int bench_rmsnorm(void) {
  const size_t tensor_shape[] = {512, 1024};
  const size_t gamma_shape[] = {1024};
  const size_t runs = 50;
  const float eps = 0.00001f;
  tt_tensor input = {0};
  tt_tensor gamma = {0};
  tt_tensor out = {0};
  int status = 1;

  if (tt_tensor_create(&input, 2, tensor_shape) != 0 ||
      tt_tensor_create(&gamma, 1, gamma_shape) != 0 ||
      tt_tensor_create(&out, 2, tensor_shape) != 0) {
    fprintf(stderr, "failed to create tensors for RMSNorm benchmark\n");
    goto cleanup;
  }

  fill_repeating(&input, 0.1f);
  fill_repeating(&gamma, 0.01f);

  // warming up once before timed runs
  if (tt_tensor_rmsnorm_last_dim(&input, &gamma, &out, eps) != 0) {
    fprintf(stderr, "RMSNorm warmup failed\n");
    goto cleanup;
  }

  double start = now_seconds();
  for (size_t run = 0; run < runs; ++run) {
    if (tt_tensor_rmsnorm_last_dim(&input, &gamma, &out, eps) != 0) {
      fprintf(stderr, "RMSNorm benchmark failed\n");
      goto cleanup;
    }
  }
  double elapsed = now_seconds() - start;
  double sum = checksum(&out);
  bench_sink = sum;

  printf("RMSNorm [512, 1024] last dim       runs=%zu total=%.6fs  avg=%.6fs  checksum=%.6f\n",
         runs, elapsed, elapsed / runs, sum);
  status = 0;

cleanup:
  tt_tensor_free(&input);
  tt_tensor_free(&gamma);
  tt_tensor_free(&out);
  return status;
}

int main(void) {
  printf("TinyTensor CPU benchmarks\n");
  printf("build: release flags from Makefile\n\n");

  if (bench_matmul() != 0 || bench_softmax() != 0 || bench_rmsnorm() != 0) {
    return 1;
  }

  return 0;
}
