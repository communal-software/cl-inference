# Design

Architectures are declared with `define-architecture`, a CLOS-backed macro. A child inherits from a parent and overrides only what differs.

| System | Purpose |
|---|---|
| `cl-inference` | [Weights protocol](weights.md), [architectures](architectures.md), blocks, sessions, sampling |
| `cl-inference/gguf` | [GGUF weight-file reader](gguf.md) |
| `cl-inference/quant` | Quantized dtypes as cl-tensor extensions |
| `cl-inference/client` | [Model client contract](client.md) and HTTP backends; no cl-tensor |

## Architectures

Tensor names, metadata keys and RoPE pairing live in per-format sections of an [architecture](architectures.md):

```lisp
(define-architecture llama ()
  (:format :gguf (:arch "llama") (:rope-pairing :adjacent)))

(define-architecture qwen3 (llama)
  (:format :gguf (:arch "qwen3") (:rope-pairing :neox)))
```

Kernels are pure Lisp on cl-tensor and trivial-simd; there is no inference-specific native code.
