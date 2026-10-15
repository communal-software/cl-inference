# KV caches

Sessions reuse the exact token prefix of complete prompts. Optional Q8 storage reduces retained K/V memory on CPU and MLX; attention still computes with temporary F32 keys and values.

## Choose storage

| Option | Retained K/V payload per 32 head elements | Attention |
|---|---:|---|
| `:cache-type :f32` (default) | 128 bytes | F32 |
| `:cache-type :q8` | 36 bytes | Dequantized F32 |

Cache storage is independent of model-weight quantization. Both options allocate to the session's `:n-ctx`, per layer on first use. Byte counts exclude tensor/array metadata, token history, weights and scratch. Partial groups retain one byte per element plus a four-byte scale.[^codec]

```lisp
(local:make-session model :n-ctx 4096 :cache-type :q8)
(local:make-local-backend weights :cache-type :q8 :reuse-prefix t)
(cl-inference/mlx:make-mlx-backend weights :cache-type :q8)
```

Q8 changes logits and may change sampled or greedy output. Pinned Llama 3.2 1B and Qwen3 0.6B checks require relative L2 at most `0.15` and cosine at least `0.99` against the F32 cache at the same token positions. Independent Python Q8 comparisons require maximum absolute error below `1.0` and relative L2 below `0.05`; quantization amplifies small arithmetic differences between engines. These bounds describe selected fixtures and prompts, not a general model-quality guarantee.

## Prefix matching

Message input and `generate :prompt-p t` represent a complete prompt. Matching compares rendered token IDs, so system messages, tool declarations, template options and control tokens participate in the comparison. Ordinary token-ID input appends.

```lisp
(local:generate session '(1 2 3) :prompt-p t :max-tokens 0)
(local:generate session '(1 2 3 4) :prompt-p t :max-tokens 0)
;; usage: :prompt-tokens 4, :cached-prompt-tokens 3
(local:generate session '(1 2 3 4) :prompt-p t :max-tokens 0)
;; usage: :prompt-tokens 4, :cached-prompt-tokens 3; token 4 is replayed
```

A shorter prompt or changed suffix truncates the logical cache at the common prefix. Identical prompts replay their last token to recover logits. Unmatched rows are overwritten and never attended to. Generated tokens participate in subsequent matching; an unconsumed stop ID does not.

`:reuse-prefix` defaults to `t` on sessions and backends. A generation override applies to that complete prompt. `nil` forces fresh prefill. `session-reset` forgets token history and retains buffers; `close-session` releases buffers and ends the session's lifetime.

## Ownership and interruption

Each session belongs to one model and one cache format. Sessions are caller-owned and must not be used concurrently. Serialized backends own one session and protect it with their existing lock. Per-call backends allocate independent sessions; there is no global cache or cross-session snapshot.

Token history commits after a successful forward pass. Cooperative cancellation keeps completed batches in direct sessions; a failed or unwound forward pass resets logical state. A serialized backend discards a failed or interrupted session. Prompt overflow is checked before replacing the existing prefix.

MLX retains quantized cache arrays on the device and transfers final logits to the sampler. Q8 execution evaluates each layer before releasing its temporary arrays. It also checks new rows for finite values through a device reduction; no K/V payload is copied to Lisp.

## Validation and measurements

See [cache acceptance](cache-acceptance.md) for pinned inputs, quality checks and reproducible measurements. The [multi-turn example](examples.md#multi-turn-q8) returns the usage counts for each turn.

## Limitations

| Ceiling | Tracker |
|---|---|
| CPU scalar conversion and full-prefix F32 attention scratch; Q8 can be slower than F32 | [#57](https://github.com/communal-software/cl-inference/issues/57) |
| MLX cache writes copy full-capacity arrays | [#49](https://github.com/communal-software/cl-inference/issues/49) |

[^codec]: K and V use independent groups along each token/head's final axis, after positional transforms. Each group stores signed bytes in `[-127,127]` with F32 scale `max(abs(group))/127`, rounded to nearest with ties to even. Zero groups use scale one and zero bytes. Nonzero scales have a minimum of the smallest normal F32 value; reconstruction saturates at finite F32 extrema. Non-finite input signals an error. This is a cache encoding, separate from GGUF Q8_0's F16 scales.
