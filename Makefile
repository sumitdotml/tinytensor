CC ?= cc
CFLAGS ?= -std=c11 -Wall -Wextra -Werror
INCLUDES := -Iinclude
ASAN_FLAGS := -g -O1 -fsanitize=address,undefined -fno-omit-frame-pointer
RELEASE_FLAGS := -O2

.PHONY: test asan clean

test:
	$(CC) $(CFLAGS) $(INCLUDES) $(RELEASE_FLAGS) src/tensor.c tests/test_tensor_lifecycle.c -o tensor_lifecycle.out
	./tensor_lifecycle.out
	$(CC) $(CFLAGS) $(INCLUDES) $(RELEASE_FLAGS) src/tensor.c tests/test_tensor_access.c -o tensor_access.out
	./tensor_access.out
	$(CC) $(CFLAGS) $(INCLUDES) $(RELEASE_FLAGS) src/tensor.c tests/test_tensor_ops.c -o tensor_ops.out
	./tensor_ops.out
	$(CC) $(CFLAGS) $(INCLUDES) $(RELEASE_FLAGS) src/tensor.c tests/test_tensor_matmul.c -o tensor_matmul.out
	./tensor_matmul.out

asan:
	$(CC) $(CFLAGS) $(INCLUDES) $(ASAN_FLAGS) src/tensor.c tests/test_tensor_lifecycle.c -o tensor_lifecycle.out
	./tensor_lifecycle.out
	$(CC) $(CFLAGS) $(INCLUDES) $(ASAN_FLAGS) src/tensor.c tests/test_tensor_access.c -o tensor_access.out
	./tensor_access.out
	$(CC) $(CFLAGS) $(INCLUDES) $(ASAN_FLAGS) src/tensor.c tests/test_tensor_ops.c -o tensor_ops.out
	./tensor_ops.out
	$(CC) $(CFLAGS) $(INCLUDES) $(ASAN_FLAGS) src/tensor.c tests/test_tensor_matmul.c -o tensor_matmul.out
	./tensor_matmul.out

clean:
	rm -rf *.out *.o *.dSYM
