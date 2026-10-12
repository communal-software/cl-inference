# Quantized weights

Load `cl-inference/quant` and Q8_0 tensors read from a GGUF file as a `:q8_0` tensor over the mapped file, with no copy.

```lisp
(asdf:load-system "cl-inference/quant")
(local:with-weights (w "Llama-3.2-1B-Instruct-Q8_0.gguf")
  (let ((model (local:load-model w)))
    (local:model-forward model '(128000 791 6864))))
```

Blocks, sessions and `linear` need no change: a `:q8_0` weight multiplies like any other.[^dispatch]

## `:q8_0` tensors

| Operation | Behaviour |
|---|---|
| `(ct:matmul x (ct:transpose weight))` | `x` an f32 matrix, `weight` a `:q8_0` matrix of `(out in)`; result f32 |
| `ct:take` | Rows stay `:q8_0` |
| `ct:astype tensor :f32`, `ct:dequantize` | f32 copy of the whole tensor, for small tensors and tests |
| Anything else | `unsupported-operation`; quantized tensors never upcast implicitly |

A view is whole rows of 32-element blocks, or the transpose of one. Views that cut a block, or start inside one, signal an error.

## Storage

| Function | Returns |
|---|---|
| `make-q8-0-storage bytes` | Storage over an `(unsigned-byte 8)` vector or trivial-simd view of whole 34-byte blocks |
| `q8-0-storage-bytes storage` | The bytes behind it |

Each block is an f16 scale and 32 signed bytes, as GGUF stores it. A tensor from `weights-tensor` points into the file mapping and is valid until the weights close.

## Accuracy

Weights dequantize exactly; the products accumulate in f32. llama.cpp also quantizes its activations to Q8_0, so layer outputs differ from its own by up to 0.03 relative error, less than its Q8_0 differs from its F16 at every layer.[^acceptance]

[^dispatch]: `linear` sends half-precision weights through [row tiles](blocks.md) and everything else to `ct:matmul`. The `:q8_0` dtype resolves that product to trivial-simd's [`q8-0-matmul!`](https://github.com/communal-software/trivial-simd/blob/trunk/docs/q8-matmul.md), a native kernel that reads the blocks in place.
[^acceptance]: See [testing](testing.md#acceptance) for the measured drift against llama.cpp.
