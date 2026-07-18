# tinytensor

A tiny C tensor library for learning ML runtime internals.

For now: `float32` only (for C), naive CUDA only (no `cudaDeviceSynchronize()` and so on), keeping
it tiny.

## Current Status

Just bootstrapped with a few helpers. Starting off from here :)

Doing this for mainly learning to write ML internals in C and ultimately CUDA. Will take a
long time until I become good at distributed ML, but hey gotta start somewhere!

## Running

Local tests and benchmarks save their binaries under the ignored `build/` directory:

```sh
make test
make asan
make bench
```

We can run the CUDA kernels on real GPU playgrounds via Modal. For this, we install Modal with
`uv tool install modal`, run `modal setup`, then run the CUDA playground. It defaults to an
RTX PRO 6000:

```sh
make modal
make modal SRC=cuda_playground/cuda_add.cu
make modal SRC=cuda_playground/cuda_add.cu GPU=L40S
```

Modal compiles and runs the CUDA binary inside its remote container. We can check active containers
and today's cost with:

```sh
modal container list
modal environment billing report --for today --show-resources
```

## Roadmap

`C memory -> CPU tensors -> PMPP -> CUDA tensor kernels`

1. CPU tensor core
   - create/free, shape validation, row-major strides
   - fill, flat get/set, multidimensional get/set
   - elementwise add/mul
   - sum and max reductions
   - matmul

2. ML-oriented CPU ops
   - softmax
   - RMSNorm or LayerNorm
   - embedding lookup
   - tiny transformer forward-pass pieces

3. Benchmarks
   - correctness checks and sanitizer builds first
   - release timing after that
   - notes on layout, cache behavior, and bottlenecks

4. CUDA backend
   - elementwise kernels
   - reductions
   - matmul experiments
   - compare selected kernels with vendor-library baselines

## Future Considerations

- more dtypes beyond `float32`
- views without copying
- broadcasting rules
- autograd or manual backward passes
- Python bindings
- more flexible dynamic shape metadata
