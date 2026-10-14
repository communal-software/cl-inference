# Quantized weights

Load `cl-inference/quant` to read Q8_0, Q4_0, Q4_K and Q6_K tensors directly from mapped GGUF files. A model can mix these dtypes with floating-point tensors.

```lisp
(asdf:load-system "cl-inference/quant")
(cl-inference/local:with-weights (weights "Qwen3-0.6B-Q4_K_M.gguf")
  (let ((model (cl-inference/local:load-model weights)))
    (unwind-protect
         (cl-inference/local:model-forward model '(791 6864))
      (cl-inference/local:close-model model))))
```

Blocks, sessions and `linear` use the same interfaces for each dtype.[^dispatch]

## Supported operations

| Operation | Behavior |
|---|---|
| `(ct:matmul x (ct:transpose weight))` | F32 matrix activations and a packed `(out in)` weight; F32 result |
| `ct:take` | Rows retain the packed dtype, including duplicate indices |
| `ct:copy-tensor` | Copies packed bytes without upcasting |
| `ct:astype tensor :f32`, `ct:dequantize` | Explicit F32 copy, including transposed views |
| Other operations or conversion targets | `unsupported-operation`; no implicit upcast |

`ct` denotes the `cl-tensor` package. Views contain whole contiguous rows of blocks, or their matrix transpose. Offsets align to blocks; views cutting blocks are rejected. The Q4/Q6 products also accept weight views starting at a whole-row offset.

## Storage

| Dtype | Elements/block | Bytes/block | Storage constructor |
|---|---:|---:|---|
| `:q8_0` | 32 | 34 | `make-q8-0-storage` |
| `:q4_0` | 32 | 18 | `make-q4-0-storage` |
| `:q4_k` | 256 | 144 | `make-q4-k-storage` |
| `:q6_k` | 256 | 210 | `make-q6-k-storage` |

Constructors in `cl-inference/quant` wrap unsigned-byte-8 vectors or trivial-simd byte views containing whole blocks. Each format exports its dtype class, storage class and byte accessor, such as `q4-k`, `q4-k-storage` and `q4-k-storage-bytes`. Q4/Q6 require simple vectors or byte views and reject other storage types.

Storage follows ggml's block layouts.[^layout] Non-finite scales signal errors in Q4/Q6 conversion and both kernel paths.

Mapped tensor storage is borrowed and remains valid only while the weights are open. A Q4_K_M file uses a mixture of Q4_K and Q6_K tensors; the suffix describes a model recipe rather than one tensor dtype.

## Execution and accuracy

Packed kernels decode only a block at a time, reuse it across activation rows, and produce F32 products. Native ARM64 execution uses NEON; other hosts use scalar native kernels. A Lisp fallback runs when the native symbol is absent or the Lisp backend is selected. Neither path expands the complete weight matrix.

[Q4/Q6 acceptance](quant-acceptance.md) compares real models with independently dequantized F32 llama.cpp execution. Native packed llama.cpp execution additionally quantizes activations and can differ more. Q8_0 retains its existing native-oracle bounds.[^q8]

## Limitations

For existing Q8_0 prefill performance restrictions, see [Blocks](blocks.md#limitations).

[^dispatch]: `linear` selects encoded matmul or bounded tiles for [half-precision weights](blocks.md#half-precision). Packed weights go to `ct:matmul`, whose extension dtype selects a `trivial-simd` kernel. Native code lives in that dependency; `cl-inference` runtime code is Lisp.
[^layout]: Q4_0 stores an F16 scale and two halves of biased four-bit values. Q4_K adds packed six-bit subscales and minima. Q6_K combines low nibbles, upper two bits and signed subscales. K formats use 256-element superblocks.
[^q8]: See [Model acceptance](acceptance.md) for the existing Q8_0 numerical comparisons.
