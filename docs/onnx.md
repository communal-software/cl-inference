# ONNX backend

`cl-inference/onnx` runs a local ONNX decoder through [cl-onnx](https://github.com/communal-software/cl-onnx) and the existing [client contract](client.md). The verified model is SmolLM2 135M Instruct FP32 on CPU, using SBCL on macOS ARM64.

## Setup and loading

Build cl-onnx's C bridge against ONNX Runtime 1.31.0 and register its ASDF system as described in its [installation guide](https://github.com/communal-software/cl-onnx/blob/trunk/docs/install.md). This repository contains only Lisp integration code; Python is used only for development references and fixture preparation.

```lisp
(asdf:load-system "cl-inference/onnx")

(cl-inference/onnx:with-onnx-backend
    (backend #p"models/SmolLM2/" :graph "onnx/model.onnx")
  (cl-inference:complete backend
                        :messages '((:role :user :content "Hello!"))
                        :temperature 0 :max-tokens 32
                        :stream #'print))
```

The directory contains `config.json`, `tokenizer.json`, `tokenizer_config.json`, the chosen graph and any external tensor data referenced by that graph. The graph filename is explicit, relative to this directory, and must resolve inside it. Model loading does not download files.

| Constructor option | Default | Meaning |
|---|---|---|
| `:graph` | Required | Relative `.onnx` filename; no automatic precision selection |
| `:name` | NIL | Name for optional explicit `register-backend` use |
| `:n-ctx` | Model context, capped at 4096 | Prompt plus generated-token capacity |
| `:n-batch` | 512 | Maximum tokens in each prefill pass |

The constructor owns the runtime session and loads tokenizer/chat metadata once. `close-onnx-backend` waits for an active completion and releases native resources; repeated closing is harmless. `with-onnx-backend` closes on every exit. Requests after closing return `(:error :unavailable)`.

Request `:model` does not select another graph. Create another backend object for another model. Register a configured backend explicitly if keyword dispatch is needed; unregister or replace that registration when its lifetime ends.

## Requests and execution

Text messages, temperature, top-p, top-k, seed, token limits, stop strings, streaming, deadlines and cancellation use the [local sampling](sampling.md) and client behavior. Unknown request keys are ignored. Tool declarations, tool history and non-text blocks return `:bad-request`.

Requests serialize per backend and each request starts with empty FP32 K/V state. Waiting for the backend counts toward the deadline. K/V outputs stay in native runtime storage between decoding steps; only the requested logits rows are copied into Lisp. Replies include neutral assistant content, finish reason and token usage; cached prompt tokens are zero.

Text deltas preserve `:ref`; each backend turn emits one closing `:done`. Stop strings are withheld across token boundaries and omitted from output. The native SmolLM2 template supplies its default system message only when the first message is not a system message. It preserves message whitespace and does not execute arbitrary Jinja.[^tokenizer]

## Interruption and concurrency

The deadline covers waiting, rendering, tokenization, inference and stream callbacks. Timeout and cancellation terminate an active native run, wait for it to return, then unwind Lisp resources. Cleanup finishes before the caller receives the failed result. The closing event uses the client's bounded sink grace.[^termination]

Reentrant completion on the same backend returns `:bad-request`. Closing the backend from its own stream callback signals an error; close it after completion returns. Separate backend objects have independent runtime sessions.

## Export boundary

Supported exports contain one Llama decoder-with-past graph with dynamic sequence lengths and batch size one. Token IDs, attention masks and position IDs are I64. Logits and per-layer K/V tensors are FP32. Names follow `input_ids`, `attention_mask`, `position_ids`, `past_key_values.N.key/value`, `logits` and `present.N.key/value`.

Loading validates this signature against the HF dimensions, tokenizer vocabulary and supported SmolLM2 template. Invalid packaging and unsupported signatures signal loading errors with an explanation; inference failures become client `:backend-error` results. Loading a graph is not acceptance for every model with a matching signature. [Pinned acceptance](onnx-acceptance.md) establishes the supported artifact.

## Limitations

| Missing | Tracker |
|---|---|
| Additional exports/dtypes, accelerators, cross-request cache reuse, concurrent per-call policy and tools | [#72](https://github.com/communal-software/cl-inference/issues/72) |

[^tokenizer]: The tokenizer applies individual digit splitting followed by GPT-2 byte-level BPE. Sparse byte tokens are handled like the pinned Hugging Face tokenizer: an absent byte token is dropped and separates BPE spans. Acceptance includes control bytes, Unicode and mixed-case contractions.

[^termination]: ONNX Runtime's [`RunOptionsSetTerminate`](https://onnxruntime.ai/docs/api/c/struct_ort_api.html) supports signalling from another thread. Lisp defers unwinding while the native call owns its buffers; abort callbacks are removed before run options are released.
