# Runnable examples

Load `cl-inference/examples` to call the tutorials. They release models and backends before closing borrowed weights.

```lisp
(ql:quickload "cl-inference/examples")
(cl-inference/examples:complete-file
 "Qwen3-0.6B-Q4_K_M.gguf"
 '((:role :user :content "Hello!"))
 :max-tokens 32 :timeout 120000)
```

The result is the assistant reply plist. Failures signal client conditions, as with `complete*`.

## Entry points

| Function | Source | Result |
|---|---|---|
| `complete-file path messages &key max-tokens timeout` | [`complete.lisp`](../examples/complete.lisp) | Greedy reply; defaults: 64 tokens, 30,000 ms |
| `chat-file path prompts &key cache-type max-tokens timeout` | [`chat.lisp`](../examples/chat.lisp) | Successive greedy replies with assistant history and prefix reuse |
| `inspect-weights path` | [`weights.lisp`](../examples/weights.lisp) | Format, architecture and tensor summaries |
| `pool-executor pool run-pool` | [`executor.lisp`](../examples/executor.lisp) | Range-executor adapter for a caller-owned synchronous pool |
| `forward-observed path tokens` | [`architecture.lisp`](../examples/architecture.lisp) | Logits and attention-output shapes for an `observed-llama` fixture |

```lisp
(cl-inference/examples:inspect-weights "model.gguf")
;; => (:format :gguf :architecture "llama" :tensors (...))
```

Inspection does not build a model or dequantize tensors. Load `cl-inference/safetensors` first to inspect HF files or directories.

## Multi-turn Q8

```lisp
(cl-inference/examples:chat-file
 "Qwen3-0.6B-f16.gguf"
 '("What is the capital of France?" "Which country is that in?")
 :cache-type :q8 :max-tokens 32 :timeout 120000)
```

Each reply contains `:meta :usage :cached-prompt-tokens`. The example owns one serialized backend, appends each assistant reply to the next request, and releases the backend before closing weights.

## GGUF on Apple Silicon

`sbcl --script examples/mlx-gguf.lisp model.gguf "Say hello."` streams a completion through the MLX backend. It requires Metal, the native cl-mlx libraries and a supported model. See [GGUF on MLX](mlx-gguf.md).

## Run without downloads

```sh
sbcl --non-interactive --eval '(require :asdf)' \
  --eval '(asdf:test-system "cl-inference/examples")'
```

The examples test system creates temporary tiny GGUF fixtures, runs the model entry points and pool adapter, and checks the observed architecture against the independent model reference. Only the test system requires development test dependencies.

The architecture fixture declares `general.architecture = "observed-llama"`; its metadata keys use that prefix. See [the architecture tutorial](architectures.md#extend-from-an-application-package) for its definition and selection rules.
