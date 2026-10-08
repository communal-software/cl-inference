# Design

Architectures are declared with `define-architecture`, a CLOS-backed macro. A child inherits from a parent and overrides only what differs.

| System | Purpose |
|---|---|
| `cl-inference` | [Weights protocol](weights.md), architectures, blocks, sessions, sampling |
| `cl-inference/gguf` | [GGUF weight-file reader](gguf.md) |
| `cl-inference/quant` | Quantized dtypes as cl-tensor extensions |

## Planned shape

Tensor names, metadata keys and RoPE pairing live in per-format sections of an architecture:[^arch]

```lisp
(define-architecture llama ()
  (:format :gguf (:arch "llama") (:rope-pairing :adjacent)))

(define-architecture qwen3 (llama)
  (:format :gguf (:arch "qwen3") (:rope-pairing :neox)))
```

Kernels are pure Lisp on cl-tensor and trivial-simd; there is no inference-specific native code.

[^arch]: [cl-inference #5](https://todo.sr.ht/~takeiteasy/cl-inference/5). GGUF `llama` permutes Q/K and uses adjacent-pair RoPE, HF safetensors uses rotate-half, and Qwen uses NeoX.
