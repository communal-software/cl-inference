# GGUF on MLX

`cl-inference/mlx-gguf` runs Llama and Qwen3 GGUF models on Apple Silicon. It extends the existing MLX loader and backend; it adds no public package or loader API.

## Run

Build cl-mlx's native libraries using the [MLX setup](mlx.md#setup), including the custom-kernel bridge, then load the integration:

```lisp
(asdf:load-system "cl-inference/mlx-gguf")
(cl-inference/local:with-weights (weights #p"models/Qwen3-0.6B-Q4_K_M.gguf")
  (cl-inference/mlx:with-mlx-backend (backend weights :n-ctx 2048)
    (cl-inference:complete* backend
      :messages '((:role :user :content "Hello!"))
      :temperature 0 :max-tokens 64)))
```

A runnable streaming example accepts a model path and optional prompt:

```sh
sbcl --script examples/mlx-gguf.lisp models/Qwen3-0.6B-Q4_K_M.gguf "Say hello."
```

## Weight formats

| Tensor format | Execution |
|---|---|
| Q4_0, Q4_K, Q6_K matrices | Raw packed bytes; custom projection and selected-row embedding kernels |
| F32, F16, BF16 | Ordinary MLX arrays, including mixed floating/packed models |
| Other consumed formats | Tensor-specific load error; no conversion fallback |

Q4_K_M is a model quantization recipe, containing both Q4_K and Q6_K tensors. Q4_0 artifacts may also contain Q6_K tensors. Embeddings and tied output heads share the same imported tensor. Architecture, tokenizer, chat rendering and stop handling follow the existing [GGUF](gguf.md), [architecture](architectures.md) and [local backend](local-backend.md) contracts.

## Memory and lifetime

Each consumed tensor is copied once from the mapped GGUF payload into MLX-owned storage. Packed bytes retain their original encoding; loading neither requantizes nor expands packed matrices to floating-point matrices. The mapping and MLX copy coexist during use, so loading is not zero-copy.

Keep source weights open until the model/backend and its sessions finish. Closing weights makes the borrowed model unavailable even though its copied arrays remain owned by MLX. Close sessions, the model/backend, then weights; cleanup is idempotent and failed loads release imported arrays.

Projection kernels decode values into registers and use 128 bytes of threadgroup reduction scratch per output workgroup. Prefill and decode remain in the MLX lazy graph. Embeddings materialize only requested rows. Evaluation and final-logit readback follow the [MLX execution contract](mlx.md#execution).[^dispatch]

Custom-kernel specializations belong to the shared [bridge cache](mlx-kernels.md). `cl-mlx/gpu:clear-kernel-cache` explicitly releases cached handles; existing lazy graphs remain valid. Closing a model does not clear caches used by other models.

## Verification

```sh
sbcl --non-interactive --eval '(require :asdf)' \
  --eval '(asdf:test-system "cl-inference/mlx-gguf")'
```

Offline fixtures compare independently decoded blocks, projections, gathered rows, mixed Llama/Qwen3 models, tied heads, RoPE factors, chunked prefill, cached decode, streaming and cleanup. Run on Apple Silicon with Metal access. [Real-model acceptance](mlx-gguf-acceptance.md) requires all pinned inputs. [Measurements](mlx-gguf-performance.md) describe complete-call timings and peak memory.

## Limitations

| Capability | Status |
|---|---|
| Q8_0 and other packed weight formats | Unsupported; select Q4_0, Q4_K_M or Q6_K artifacts |
| Packed-weight reuse across prefill token rows | [#68](https://github.com/communal-software/cl-inference/issues/68) |
| Platforms and Lisp runtimes | Verified on macOS Apple Silicon with SBCL |
| GPU sampling and inference fusion | Outside this integration; sampling uses the existing CPU sampler |
| Affine quantization extensions | [MLX limitations](mlx.md#limitations) |

[^dispatch]: Single-token and batched projections use separate cached wrappers with format-specialized kernels. Each 32-thread workgroup computes one output value. Noncontiguous inputs normalize lazily through the bridge. Block alignment, payload sizes and finite block scales are checked before import.
