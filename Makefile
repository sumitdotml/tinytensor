CC ?= cc
CFLAGS ?= -std=c11 -Wall -Wextra -Werror
INCLUDES := -Iinclude
ASAN_FLAGS := -g -O1 -fsanitize=address,undefined -fno-omit-frame-pointer
RELEASE_FLAGS := -O2

.PHONY: test asan bench clean

test:
	$(CC) $(CFLAGS) $(INCLUDES) $(RELEASE_FLAGS) src/tensor.c tests/test_tensor_lifecycle.c -o tensor_lifecycle.out
	./tensor_lifecycle.out
	$(CC) $(CFLAGS) $(INCLUDES) $(RELEASE_FLAGS) src/tensor.c tests/test_tensor_access.c -o tensor_access.out
	./tensor_access.out
	$(CC) $(CFLAGS) $(INCLUDES) $(RELEASE_FLAGS) src/tensor.c tests/test_tensor_basic_ops.c -o tensor_basic_ops.out
	./tensor_basic_ops.out
	$(CC) $(CFLAGS) $(INCLUDES) $(RELEASE_FLAGS) src/tensor.c tests/test_tensor_matmul.c -o tensor_matmul.out
	./tensor_matmul.out
	$(CC) $(CFLAGS) $(INCLUDES) $(RELEASE_FLAGS) src/tensor.c tests/test_tensor_softmax.c -o tensor_softmax.out
	./tensor_softmax.out
	$(CC) $(CFLAGS) $(INCLUDES) $(RELEASE_FLAGS) src/tensor.c tests/test_tensor_rmsnorm.c -o tensor_rmsnorm.out
	./tensor_rmsnorm.out

asan:
	$(CC) $(CFLAGS) $(INCLUDES) $(ASAN_FLAGS) src/tensor.c tests/test_tensor_lifecycle.c -o tensor_lifecycle.out
	./tensor_lifecycle.out
	$(CC) $(CFLAGS) $(INCLUDES) $(ASAN_FLAGS) src/tensor.c tests/test_tensor_access.c -o tensor_access.out
	./tensor_access.out
	$(CC) $(CFLAGS) $(INCLUDES) $(ASAN_FLAGS) src/tensor.c tests/test_tensor_basic_ops.c -o tensor_basic_ops.out
	./tensor_basic_ops.out
	$(CC) $(CFLAGS) $(INCLUDES) $(ASAN_FLAGS) src/tensor.c tests/test_tensor_matmul.c -o tensor_matmul.out
	./tensor_matmul.out
	$(CC) $(CFLAGS) $(INCLUDES) $(ASAN_FLAGS) src/tensor.c tests/test_tensor_softmax.c -o tensor_softmax.out
	./tensor_softmax.out
	$(CC) $(CFLAGS) $(INCLUDES) $(ASAN_FLAGS) src/tensor.c tests/test_tensor_rmsnorm.c -o tensor_rmsnorm.out
	./tensor_rmsnorm.out

bench:
	$(CC) $(CFLAGS) $(INCLUDES) $(RELEASE_FLAGS) src/tensor.c benchmarks/bench_tensor.c -o tensor_bench.out
	./tensor_bench.out

clean:
	rm -rf *.out *.o *.dSYM
