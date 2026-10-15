# Local backend

`make-local-backend` runs a model through the [client contract](client.md). It borrows open weights and loads their model, tokenizer and chat template once.

```lisp
(asdf:load-system :cl-inference/gguf)
;; Load :cl-inference/quant as well for Q8_0, Q4_0, Q4_K or Q6_K weights.

(cl-inference/local:with-weights (weights "model.gguf")
  (let ((backend (cl-inference/local:make-local-backend weights :n-ctx 2048)))
    (cl-inference:complete backend
                          :messages '((:role :user :content "Hello!"))
                          :temperature 0 :max-tokens 64)))
```

Keep weights open until every completion finishes. The backend does not close them. Calling a backend after its weights close returns `(:error :unavailable)`.

HF model directories use the shipped [HF tokenizer reader](hf-tokenizers.md). For Apple Silicon execution and packed affine weights, use the [MLX backend](mlx.md).

## Configuration

| Constructor option | Default | Meaning |
|---|---|---|
| `:name` | `NIL` | Keyword for optional registration |
| `:n-ctx` | Model context, capped at 4096 | Maximum cached positions |
| `:n-batch` | 512 | Tokens per prefill pass |
| `:session-policy` | `:serialized` | Cache reuse and concurrency |
| `:reuse-prefix` | `t` | Reuse the session's matching rendered-token prefix |
| `:cache-type` | `:f32` | `:f32` or symmetric `:q8` K/V storage |

The constructor's weights select the model. Request `:model` does not switch models or open a file. Use another backend object for another model.

The initially registered `:local` backend is unconfigured and unavailable. Register a configured backend to use the keyword:

```lisp
(cl-inference:register-backend
 (cl-inference/local:make-local-backend weights :name :local))
(cl-inference:complete :local :messages messages)
```

Keep registration inside the weights' lifetime. `describe-backend` reports `:ready` or `:unavailable`, supported sampling parameters and the session policy.

## Memory and concurrency

| Policy | Cache | Concurrent requests |
|---|---|---|
| `:serialized` | One reusable session | Wait for the current turn; waiting counts toward timeout |
| `:per-call` | Fresh session per call | Run independently; each active call allocates its own cache |

Both modes share model weights and render the complete request. Serialized mode reuses its session's matching token prefix after successful turns and discards the session after a failed or interrupted turn. Per-call sessions share no cache. Unrelated prompts safely overwrite the unmatched suffix; `:reuse-prefix nil` disables matching. See [cache ownership and matching](kv-cache.md).

Use `:serialized` for lower memory use. `:n-ctx` and `:n-batch` control cache capacity and prefill size. A nested call from the same serialized backend's delta sink returns a bad request; use `:per-call` for that pattern.

## Requests and replies

The backend accepts text messages, tools, `:temperature`, `:top-p`, `:top-k`, `:seed`, `:max-tokens`, `:stop`, `:timeout`, `:cancel`, `:stream` and `:ref`. Unknown keys are ignored.

Sampling defaults are [the sampler's defaults](sampling.md): temperature 1, top-p 1, no top-k filter and a random seed. Each completion creates a fresh sampler. Without `:max-tokens`, generation runs until a stop or the context limit. Stop strings are a list of non-empty strings.

Replies carry the client's assistant content, tool calls, `:done t`, and `:meta` with `:finish-reason` and token usage. Valid tool calls normally finish with `:tool-calls`; a token or context limit keeps `:length`.

## Tools and reasoning

| Template | Generated output |
|---|---|
| Plain ChatML | Text; tool requests and tool history are unsupported |
| Qwen2.5 and Hermes | Text and validated JSON tool calls |
| Qwen3 and QwQ | JSON tool calls and reasoning |
| Qwen3-Coder and Qwen3.5 | Function/parameter calls; Qwen3.5 reasoning |
| R1 distills | DeepSeek tool groups and reasoning |
| Llama 3.0 | Text only |
| Llama 3.1/3.2 | Declared JSON calls, tagged Python; explicit builtin mode for 3.1 |

Parsed reasoning is kept in `:meta :reasoning` and omitted from text deltas. An unfinished thinking block at a length limit remains reasoning. Direct `generate` calls return raw text; use this backend for parsed replies.

The local request forwards `:date`, `:enable-thinking`, `:builtin-tools` and `:tools-in-user-message` consistently to rendering and parsing. [Family options](chat-families.md) describe defaults and restrictions.

Llama Python output requires a declared tool:

```lisp
:tools '((:name :python :summary "Run Python"
          :params ((:code string :required t))))
```

The reply carries `(:name :python :arguments (:code "..."))` with an ID and schema. The backend never executes tools.

Malformed or truncated calls, missing arguments, and undeclared tool names return `:backend-error`. Unsupported prompt features return `:bad-request` with an explanation.

## Streaming and deadlines

Text streams through neutral `:text-delta` events. Each tool call is buffered until its JSON validates, then emitted as one `:tool-call-delta` with complete argument JSON. Tool syntax and reasoning are omitted from text events.

Every backend turn emits one closing `:done` when a sink is supplied. Its reason matches the final result. The deadline covers waiting for a session, prompt rendering, generation, parsing and sink callbacks; the closing event uses the [client's sink grace](client.md#events). Cancellation releases session ownership, and the next request can run normally.

## Limitations

| Missing | Tracker |
|---|---|
| Q8 conversion speed and full-prefix attention scratch | [#57](https://github.com/communal-software/cl-inference/issues/57) |
