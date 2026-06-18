CC ?= cc
CFLAGS ?= -std=c11 -Wall -Wextra -Werror
INCLUDES := -Iinclude
ASAN_FLAGS := -g -O1 -fsanitize=address,undefined -fno-omit-frame-pointer
RELEASE_FLAGS := -O2

.PHONY: test asan clean

test:
	$(CC) $(CFLAGS) $(INCLUDES) $(RELEASE_FLAGS) src/tensor.c tests/test_tensor_lifecycle.c -o tensor_lifecycle.out
	./tensor_lifecycle.out

asan:
	$(CC) $(CFLAGS) $(INCLUDES) $(ASAN_FLAGS) src/tensor.c tests/test_tensor_lifecycle.c -o tensor_lifecycle.out
	./tensor_lifecycle.out

clean:
	rm -rf *.out *.o *.dSYM
