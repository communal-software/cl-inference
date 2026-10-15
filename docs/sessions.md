# Sessions

A session holds a model's KV cache and position. `generate` runs one turn on it: feed a prompt, then sample until the reply ends.

```lisp
(local:with-weights (w "Llama-3.2-1B-Instruct-f16.gguf")
  (let* ((model (local:load-model w))
         (session (local:make-session model
                                      :tokenizer (local:load-tokenizer w)
                                      :chat-template (local:load-chat-template w))))
    (local:generate session '((:role :user :content "Hello!")) :max-tokens 64)))
;; => "Hi! How can I help?", :stop, (9906 0 ...), (:prompt-tokens 14 :completion-tokens 8 :total-tokens 22)
```

Bind `ct:*executor*` around session calls for [parallel CPU inference](parallel-inference.md). The caller owns the pool; session ownership and cancellation checkpoints remain the same.

## Functions

| Function | Does |
|---|---|
| `make-session model &key tokenizer chat-template n-ctx n-batch` | A session with an empty cache |
| `generate session input &key sampler max-tokens stop stop-ids timeout cancel sink ref special` | One turn; returns `(values text finish-reason token-ids usage)` |
| `session-prefill session tokens &key check` | Feeds tokens `n-batch` at a time; returns the last position's logits |
| `session-reset session` | Position 0; the cache buffers are kept |
| `session-position session` | Tokens in the cache |
| `greedy-sampler logits` | The default sampler: the largest logit's id |
| `make-sampler &key temperature top-k top-p seed` | A [sampler](sampling.md) |

## Sizing

| Option | Default | Meaning |
|---|---|---|
| `:n-ctx` | The model's `context-length` hparam, at most 4096; 4096 without one | Positions the cache holds. Its K and V buffers are allocated per layer on first use |
| `:n-batch` | 512 | Tokens per prefill pass |

## Turns

`input` is token ids, appended after the session's position, or messages, which reset the session and are rendered by its chat template.

A turn ends with the first of:

| `finish-reason` | When |
|---|---|
| `:stop` | The sampler returns a stop id, or the text reaches a `:stop` string |
| `:length` | `:max-tokens` tokens were generated, or the context is full |
| `(:error :timeout)` | `:timeout` milliseconds, covering prefill and decoding, pass; default 30000 |
| `(:error :cancelled)` | The `:cancel` token fired |

The deadline and token are checked before each prefill pass and each decoding step.

`:stop-ids` defaults to `chat-template-stop-ids` when the session has a template and a tokenizer. `:stop` strings need a tokenizer; text that could begin a stop string is held back until it cannot.

`:sampler` is a [sampler](sampling.md) or a function from the `(1 vocab)` logits to a token id.[^sampler]

## Events

`:sink` receives the [client's events](client.md#events): a `:text-delta` for each released piece of text, then one `:done` whose `:reason` is the finish reason. `:ref` is stamped on each. Without a tokenizer there is no text and no deltas; `text` is `NIL`.

`generate :special t` retains decoded control tokens for [reply parsing](chat-templates.md#reply-parsing). The default omits them. Stop ids never appear in text or returned ids.

## Cache state

Every returned token is in the cache when `generate` returns, so token input continues the turn:

```lisp
(local:generate session '(1 2 3) :max-tokens 2)   ; => ... (7 9)
(local:generate session '(4))                     ; continues after 1 2 3 7 9 4
```

A stop id is neither returned nor cached. When the context fills, `generate` ends with `:length` and every returned token is cached.

## Errors

| Condition | Signalled when |
|---|---|
| `context-overflow` | The input does not fit after the session's position |
| `session-error` | Empty input, messages without a template and tokenizer, or stop strings without a tokenizer or that are empty |

`context-overflow` is a `session-error`; `session-error-message` returns the text.

## Limitations

| Missing | Tracker |
|---|---|
| Reusing the cache across messages or sessions | [#16](https://github.com/communal-software/cl-inference/issues/16) |
| Bulk copy of new keys and values into the cache | [#39](https://github.com/communal-software/cl-inference/issues/39) |
| Quantized cache | [#18](https://github.com/communal-software/cl-inference/issues/18) |

[^sampler]: A stop id is checked on the sampler's choice, so a sampler that returns one ends the turn.
