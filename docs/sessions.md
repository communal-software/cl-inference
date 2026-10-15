# Sessions

A session holds a model's KV cache, tokens and position. `generate` runs one turn on it; complete prompts reuse their matching token prefix. [Cache behavior](kv-cache.md) covers branching, ownership and Q8 storage.

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
| `make-session model &key tokenizer chat-template n-ctx n-batch cache-type reuse-prefix` | A session with an empty cache |
| `generate session input &key sampler max-tokens stop stop-ids timeout cancel sink ref special prompt-p reuse-prefix` | One turn; returns `(values text finish-reason token-ids usage)` |
| `session-prefill session tokens &key check` | Feeds tokens `n-batch` at a time; returns the last position's logits |
| `session-reset session` | Position 0; the cache buffers are kept |
| `session-position session` | Tokens in the cache |
| `session-cache-type session` | `:f32` or `:q8` |
| `session-reuse-prefix session` | Default prefix-reuse policy |
| `close-session session` | Releases cache storage; the session is no longer usable |
| `greedy-sampler logits` | The default sampler: the largest logit's id |
| `make-sampler &key temperature top-k top-p seed` | A [sampler](sampling.md) |

## Sizing

| Option | Default | Meaning |
|---|---|---|
| `:n-ctx` | The model's `context-length` hparam, at most 4096; 4096 without one | Positions the cache holds. Its K and V buffers are allocated per layer on first use |
| `:n-batch` | 512 | Tokens per prefill pass |
| `:cache-type` | `:f32` | Floating-point or symmetric Q8 K/V storage; independent of weight quantization |
| `:reuse-prefix` | `t` | Reuse matching tokens for complete prompts |

## Turns

`input` is token ids, appended after the session's position, or messages rendered by its chat template. Messages and token IDs passed with `:prompt-p t` replace the logical prompt and reuse its exact matching prefix. `:reuse-prefix nil` forces fresh prefill for a complete prompt. Plain token-ID input continues to append.

Usage reports the full `:prompt-tokens`, `:completion-tokens` and `:total-tokens`, plus `:cached-prompt-tokens` for positions whose prefill was skipped. An identical or shorter cached prompt replays its final token for logits; that position does not count as cached.

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
| Bulk copy of new keys and values into the cache | [#39](https://github.com/communal-software/cl-inference/issues/39) |
| Q8 conversion speed and full-prefix attention scratch | [#57](https://github.com/communal-software/cl-inference/issues/57) |

[^sampler]: A stop id is checked on the sampler's choice, so a sampler that returns one ends the turn.
