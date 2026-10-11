CC ?= cc
NVCC ?= nvcc
CFLAGS ?= -std=c11 -Wall -Wextra -Werror
CUDAFLAGS ?= -std=c++17 -O2
INCLUDES := -Iinclude
ASAN_FLAGS := -g -O1 -fsanitize=address,undefined -fno-omit-frame-pointer
RELEASE_FLAGS := -O2
BUILD_DIR := build
SRC ?= cuda_playground/cuda_add.cu
CUDA_HELPERS ?= include/cuda_helpers.h
OUT := $(basename $(notdir $(SRC))).out
GPU ?= RTX-PRO-6000
MODAL_IMAGE ?= nvidia/cuda:12.8.1-devel-ubuntu24.04

.PHONY: test asan bench cuda-test modal clean

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

test: | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) $(RELEASE_FLAGS) src/tensor.c tests/test_tensor_lifecycle.c -o $(BUILD_DIR)/tensor_lifecycle.out
	./$(BUILD_DIR)/tensor_lifecycle.out
	$(CC) $(CFLAGS) $(INCLUDES) $(RELEASE_FLAGS) src/tensor.c tests/test_tensor_access.c -o $(BUILD_DIR)/tensor_access.out
	./$(BUILD_DIR)/tensor_access.out
	$(CC) $(CFLAGS) $(INCLUDES) $(RELEASE_FLAGS) src/tensor.c tests/test_tensor_basic_ops.c -o $(BUILD_DIR)/tensor_basic_ops.out
	./$(BUILD_DIR)/tensor_basic_ops.out
	$(CC) $(CFLAGS) $(INCLUDES) $(RELEASE_FLAGS) src/tensor.c tests/test_tensor_matmul.c -o $(BUILD_DIR)/tensor_matmul.out
	./$(BUILD_DIR)/tensor_matmul.out
	$(CC) $(CFLAGS) $(INCLUDES) $(RELEASE_FLAGS) src/tensor.c tests/test_tensor_softmax.c -o $(BUILD_DIR)/tensor_softmax.out
	./$(BUILD_DIR)/tensor_softmax.out
	$(CC) $(CFLAGS) $(INCLUDES) $(RELEASE_FLAGS) src/tensor.c tests/test_tensor_rmsnorm.c -o $(BUILD_DIR)/tensor_rmsnorm.out
	./$(BUILD_DIR)/tensor_rmsnorm.out

asan: | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) $(ASAN_FLAGS) src/tensor.c tests/test_tensor_lifecycle.c -o $(BUILD_DIR)/tensor_lifecycle.out
	./$(BUILD_DIR)/tensor_lifecycle.out
	$(CC) $(CFLAGS) $(INCLUDES) $(ASAN_FLAGS) src/tensor.c tests/test_tensor_access.c -o $(BUILD_DIR)/tensor_access.out
	./$(BUILD_DIR)/tensor_access.out
	$(CC) $(CFLAGS) $(INCLUDES) $(ASAN_FLAGS) src/tensor.c tests/test_tensor_basic_ops.c -o $(BUILD_DIR)/tensor_basic_ops.out
	./$(BUILD_DIR)/tensor_basic_ops.out
	$(CC) $(CFLAGS) $(INCLUDES) $(ASAN_FLAGS) src/tensor.c tests/test_tensor_matmul.c -o $(BUILD_DIR)/tensor_matmul.out
	./$(BUILD_DIR)/tensor_matmul.out
	$(CC) $(CFLAGS) $(INCLUDES) $(ASAN_FLAGS) src/tensor.c tests/test_tensor_softmax.c -o $(BUILD_DIR)/tensor_softmax.out
	./$(BUILD_DIR)/tensor_softmax.out
	$(CC) $(CFLAGS) $(INCLUDES) $(ASAN_FLAGS) src/tensor.c tests/test_tensor_rmsnorm.c -o $(BUILD_DIR)/tensor_rmsnorm.out
	./$(BUILD_DIR)/tensor_rmsnorm.out

bench: | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) $(RELEASE_FLAGS) src/tensor.c benchmarks/bench_tensor.c -o $(BUILD_DIR)/tensor_bench.out
	./$(BUILD_DIR)/tensor_bench.out

cuda-test: | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) $(RELEASE_FLAGS) -c src/tensor.c -o $(BUILD_DIR)/tensor.o
	$(NVCC) $(CUDAFLAGS) $(INCLUDES) $(BUILD_DIR)/tensor.o src/tensor_cuda.cu tests/test_tensor_cuda_add.cu -o $(BUILD_DIR)/tensor_cuda_add.out
	./$(BUILD_DIR)/tensor_cuda_add.out
	$(NVCC) $(CUDAFLAGS) $(INCLUDES) $(BUILD_DIR)/tensor.o src/tensor_cuda.cu tests/test_tensor_cuda_matmul.cu -o $(BUILD_DIR)/tensor_cuda_matmul.out
	./$(BUILD_DIR)/tensor_cuda_matmul.out

modal:
	modal shell --gpu $(GPU) --image $(MODAL_IMAGE) --add-local $(SRC) --add-local $(CUDA_HELPERS) -c 'nvcc $(CUDAFLAGS) /mnt/$(notdir $(SRC)) -o /tmp/$(OUT) && /tmp/$(OUT)'

clean:
	rm -rf $(BUILD_DIR) *.out *.o *.dSYM
