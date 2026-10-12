# Design

`cl-inference` is a model client: one `complete` contract for hosted and local models. `cl-inference/local` runs a model in the same image and plugs in as one more backend, so a consumer switches between them by naming a different backend.

```lisp
(complete :ollama :messages msgs)   ; hosted or on another host
(complete :local  :messages msgs)   ; in this image
```

| System | Purpose |
|---|---|
| `cl-inference` | [Model client contract](client.md) and HTTP backends; no cl-tensor |
| `cl-inference/local` | `:local` backend, [weights protocol](weights.md), [architectures](architectures.md), [tokenizers](tokenizers.md), [chat templates](chat-templates.md), [blocks](blocks.md), [sessions](sessions.md), [sampling](sampling.md) |
| `cl-inference/gguf` | [GGUF weight-file reader](gguf.md) |
| `cl-inference/quant` | Quantized dtypes as cl-tensor extensions |

## Architectures

Architectures are declared with `define-architecture`, a CLOS-backed macro. A child inherits from a parent and overrides only what differs. Tensor names, metadata keys and RoPE pairing live in per-format sections of an [architecture](architectures.md):

```lisp
(define-architecture llama ()
  (:format :gguf (:arch "llama") (:rope-pairing :adjacent)))

(define-architecture qwen3 (llama)
  (:format :gguf (:arch "qwen3") (:rope-pairing :neox)))
```

Kernels are pure Lisp on cl-tensor and trivial-simd; there is no inference-specific native code.

## Tokenizers

Tokenizers are declared the same way with `define-tokenizer`, selected by the file's `tokenizer.ggml.model` and `tokenizer.ggml.pre` rather than its architecture:

```lisp
(define-tokenizer qwen2 (llama-bpe)
  (:pre-tokenizer (digits :max 1))
  (:format :gguf (:pre "qwen2")))
```
