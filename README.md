# tinytensor

A tiny C tensor library for learning ML runtime internals.

For now: CPU-only, `float32` only, keeping it tiny.

## Current Status

Just bootstrapped with a few helpers. Starting off from here :)

Doing this for mainly learning to write ML internals in C and ultimately CUDA. Will take a
long time until I become good at distributed ML, but hey gotta start somewhere!

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
