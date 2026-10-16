# MLX custom kernels

`cl-mlx/gpu` runs cl-gpu DSL kernels as lazy operations over MLX arrays. It is an optional system in the cl-mlx repository; the standard inference backend does not load it automatically.

## Setup and example

Rebuild cl-mlx's native shim, then load the bridge:

```lisp
(asdf:load-system "cl-mlx/gpu")
```

Use `cl-gpu:define-kernel` for the DSL body and `cl-mlx/gpu:define-kernel` for a wrapper declaring output shapes and grid/group expressions. Wrappers accept MLX arrays and typed scalar/vector values, and return a list of lazy MLX arrays.

The [cl-mlx API guide](https://github.com/communal-software/cl-mlx/blob/trunk/docs/gpu.md) documents argument layouts, initialized outputs, integer/boolean templates and ownership. Its [runnable example](https://github.com/communal-software/cl-mlx/blob/trunk/examples/custom-kernels.lisp) adds packed three-lane vectors, composes the result with ordinary MLX addition and evaluates once.

Contiguous input arrays remain in the MLX graph without a bridge copy. Noncontiguous inputs normalize lazily on the GPU. Outputs follow cl-mlx's array scope/release rules, and `clear-kernel-cache` releases cached native kernel handles independently of existing output graphs.[^runtime]

## Verification

```sh
sbcl --non-interactive --eval '(require :asdf)' \
  --eval '(asdf:test-system "cl-mlx/gpu")' \
  --eval '(asdf:test-system "cl-inference/mlx/tests")'
```

Run on macOS Apple Silicon with Metal access. Bridge fixtures cover scalar/vector layouts, shared memory, atomic initialization, lazy composition, template specialization, kernel redefinition and cleanup. Existing inference fixtures retain the ordinary MLX backend path.

## Limitations

| Integration | Status |
|---|---|
| GGUF Q4_0/Q4_K/Q6_K model execution | [Optional GGUF integration](mlx-gguf.md) |
| GPU sampling and inference fusion | Separate integrations; standard MLX inference uses the existing CPU sampler |
| MLX affine widths and modes | [#50](https://github.com/communal-software/cl-inference/issues/50) |

The bridge rejects `precise-math`, shared/private pointer parameters, read-only atomic inputs and mixed atomic/non-atomic outputs. Its [API limitations](https://github.com/communal-software/cl-mlx/blob/trunk/docs/gpu.md#limitations) describe the supported subset.

[^runtime]: The bridge uses cl-gpu's embeddable Metal source and MLX's custom-kernel API. It uses neither the cl-gpu Metal runtime nor cl-tensor GPU storage. Native dispatch uses the calling thread's MLX GPU stream.
