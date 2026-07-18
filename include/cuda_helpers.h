#ifndef TINYTENSOR_CUDA_HELPERS_H
#define TINYTENSOR_CUDA_HELPERS_H

#include <cuda_runtime.h>
#include <stdio.h>

static inline int tt_cuda_print_device_info(void) {
  int device = 0;
  cudaDeviceProp prop;

  if (cudaGetDevice(&device) != cudaSuccess ||
      cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
    return 1;
  }

  printf("GPU: %s\n", prop.name);
  printf("compute capability: %d.%d\n", prop.major, prop.minor);
  return 0;
}

#endif
