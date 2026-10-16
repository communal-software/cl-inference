# MLX backend

`cl-inference/mlx` runs local Llama and Qwen3 models on Apple Silicon through [cl-mlx](https://github.com/communal-software/cl-mlx). It uses the same request, reply and streaming contract as the other backends.

## Setup

Build cl-mlx's native libraries with its [installation instructions](https://github.com/communal-software/cl-mlx/blob/trunk/docs/install.md), then load the optional system:

Q8 cache support requires the [symmetric Q8 bindings and finite reconstruction](https://github.com/communal-software/cl-mlx/commit/e9d7d40) and a rebuilt native shim.

```lisp
(asdf:load-system "cl-inference/mlx")
(cl-inference/local:with-weights (weights #p"models/Qwen3-0.6B/")
  (cl-inference/mlx:with-mlx-backend (backend weights :n-ctx 2048)
    (cl-inference:complete backend
      :messages '((:role :user :content "Hello!"))
      :temperature 0 :max-tokens 64)))
```

[GGUF models](mlx-gguf.md) use the optional `cl-inference/mlx-gguf` integration.

The model directory contains Safetensors weights, `config.json`, `tokenizer.json` and `tokenizer_config.json`. Indexed shards use the existing [Safetensors loader](safetensors.md).

## Supported models

| Capability | Coverage |
|---|---|
| Architectures | Shipped Llama and Qwen3 declarations |
| Float storage | F32, F16 and BF16 |
| Quantization | MLX affine 4-bit; group sizes 32, 64 and 128 |
| Attention | Causal GQA; Qwen3 Q/K normalization |
| RoPE | Ordinary and Llama 3 scaling; HF split-half pairing |
| Tokenizer and chat | [Shipped HF byte-level BPE pipelines](hf-tokenizers.md) and existing renderers |
| Real-model acceptance | Llama 3.2 1B Instruct and Qwen3 0.6B, BF16 and affine 4-bit |

Quantized projections and tied output heads use packed matmul. Quantized embeddings dequantize only selected rows. Computation, scales and quantization biases use F32; full packed matrices stay packed. Caches use F32 by default or optional [symmetric Q8 storage](kv-cache.md). Tiny fixtures cover both architectures, mixed weights and tied output heads.

## Configuration and lifetime

`make-mlx-backend` borrows open weights. Its keyword arguments are `:name` (default `:mlx`), `:n-ctx`, `:n-batch` (default 512), `:session-policy` (default `:serialized`), `:reuse-prefix` (default `t`) and `:cache-type` (`:f32` or `:q8`, default `:f32`). Sampling, stops, tools, reasoning, deadlines and cancellation follow the [local backend](local-backend.md).

Keep weights and the backend open until every completion finishes. `with-mlx-backend` releases its model and cache; `close-mlx-backend` provides explicit, idempotent cleanup and does not close weights. Calls after closing either return `(:error :unavailable)`.

The registered `:mlx` backend is initially unconfigured. Register a configured backend within its weights and backend lifetime to use it by keyword.

Direct token-ID inference uses `load-mlx-model` with the shared `make-session`, `session-prefill` and `generate` functions. Close sessions with `close-session` and the model with `close-model`.

## Execution

A floating-cache forward pass constructs one lazy graph and evaluates logits and updated caches together. Each decode step has one explicit evaluation; prefill evaluates once per batch. Q8 caches also evaluate each layer to release dequantization scratch. Only final logits move to the existing CPU sampler. Temporary array handles are released after each pass; model and cache arrays remain owned by their objects.[^execution]

The [acceptance instructions](mlx-acceptance.md) describe the pinned reference and required checks.

## Custom kernels

The optional `cl-mlx/gpu` system runs cl-gpu DSL kernels as lazy MLX operations. [Custom kernel integration](mlx-kernels.md) covers loading, generated wrappers and the runnable example.

## Limitations

| Deferred capability | Tracker |
|---|---|
| Cache updates avoiding full-capacity memory traffic | [#49](https://github.com/communal-software/cl-inference/issues/49) |
| Q8 conversion speed and full-prefix attention scratch | [#57](https://github.com/communal-software/cl-inference/issues/57) |
| Additional quantization widths and modes | [#50](https://github.com/communal-software/cl-inference/issues/50) |
| GPU sampling and inference fusion | [Custom kernel limitations](mlx-kernels.md#limitations) |
| Other HF tokenizer pipelines and templates | [HF limitations](hf-tokenizers.md#limitations) |

Platforms other than macOS Apple Silicon are outside this backend's coverage. The verified Lisp runtime is SBCL.

[^execution]: Shared execution dispatch selects the model and cache implementation without making MLX a cl-tensor storage extension. MLX streams belong to the native calling thread. Cancellation is checked between passes; submitted native work finishes before its resources are released.
