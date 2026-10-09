# Client

`cl-inference/client` is one synchronous `complete` contract for chat models, with HTTP backends behind it. It depends on neither `cl-inference` nor cl-tensor, so HTTP-only consumers do not load the engine.

```lisp
(asdf:load-system :cl-inference/client)
(use-package :cl-inference/client)

(complete :ollama :model "llama3.2"
                  :messages '((:role :user :content "Reply with ok.")))
;; => (:ok (:role :assistant :content ((:type :text :text "ok")) :tool-calls nil
;;          :done t :meta (:finish-reason :stop :usage (...))))
```

## Calling

| Function | Answers | On failure |
|---|---|---|
| `(complete backend &rest request)` | `(:ok reply)` | `(:error reason)` |
| `(complete* backend &rest request)` | the reply plist | signals a `completion-error` |

`backend` is a keyword registered with `register-backend` or a backend object. An unregistered keyword is `(:error :unavailable)`.

## Request

A plist. Unknown keys are ignored, so a caller may offer a superset.

| Key | Meaning |
|---|---|
| `:messages` | Required. Plists with `:role` (`:system`, `:user`, `:assistant`, `:tool`), `:content`, and for tools `:tool-calls` / `:tool-call-id` |
| `:model`, `:base-url`, `:headers` | Provider data; a [provider](#providers) fills them in |
| `:tools` | Tool metadata: `:name`, `:summary`, `:params` (a [schema](#schemas)) |
| `:temperature`, `:top-p`, `:max-tokens`, `:stop`, `:seed` | Sampling; each protocol advertises the ones it renders |
| `:timeout` | Milliseconds for the whole exchange. Default 30000 |
| `:cancel` | A token from `make-cancel-token`; `(cancel token)` ends the call |
| `:stream` | A function called with each [event](#events) |
| `:ref` | Stamped on every event |

`:content` is a string or a list of blocks, `(:type :text :text "...")`.

## Reply

```lisp
(:role :assistant :content <blocks or nil> :tool-calls <calls> :done t
 :meta (:finish-reason :stop :usage (:prompt-tokens 7 ...) :id "..."))
```

A tool call is `(:id "c9" :name :tool-shell :arguments (:cmd "ls") :schema ...)`: a keyword name and an argument plist.

## Errors

| Result | Condition | Meaning |
|---|---|---|
| `:timeout` | `completion-timeout` | The deadline passed |
| `:cancelled` | `completion-cancelled` | The `:cancel` token fired |
| `:unavailable` | `completion-unavailable` | Nothing was read: refused, closed early, unknown backend |
| `(:bad-request msg)` | `completion-bad-request` | The request failed its pre-flight check |
| `(:backend-error status detail [:retry-after ms])` | `completion-backend-error` | The backend answered badly: non-2xx, malformed payload, cut stream |

`completion-backend-error` has readers `completion-backend-error-status`, `-detail` and `-retry-after`.[^retry]

## Events

A function `:stream` sink receives plists, never raw wire chunks:

| `:type` | Fields |
|---|---|
| `:text-delta` | `:ref`, `:text` |
| `:tool-call-delta` | `:ref`, `:id`, `:name`, `:arguments` (a fragment) |
| `:done` | `:ref`, `:reason` (finish reason, or the failed result) |

A turn ends with exactly one `:done`. The sink runs on the calling thread; a sink that signals is ignored.[^sink]

## Backends

| Backend | Registered as | Speaks |
|---|---|---|
| `openai-backend` | `:protocol-openai` | OpenAI-compatible chat completions, SSE streaming |
| `ollama-backend` | `:protocol-ollama` | Ollama native `/api/chat`, NDJSON streaming |
| `scripted-backend` | not registered | Canned replies, for offline tests |

`describe-backend` returns a metadata plist (`:kind`, `:name`, `:summary`, ...). `(backends :kind :provider)` lists registered names.

Writing a backend means subclassing `backend` and adding a method:

```lisp
(defmethod backend-complete ((backend my-backend) request)
  (list :ok (list :role :assistant :content (normalize-content "hi") :done t)))
```

`request` is already checked. The method honours `:cancel`, `:timeout` and `:stream`.

## Providers

A provider is data layered under each request: a protocol, base URL, authentication, model catalogue and quirks.

```lisp
(define-provider :ollama
  :protocol :protocol-ollama
  :base-url "http://127.0.0.1:11434"
  :auth :none
  :models '("llama3.2"))

(complete (make-provider :ollama :model "llama3.2") :messages ...)
```

| Key | Meaning |
|---|---|
| `:protocol` | A registered protocol or provider |
| `:base-url` | `http` or `https` URL |
| `:auth` | `:none`, `(:bearer :env "VAR")` or `(:header "name" :env "VAR")` |
| `:defaults`, `:headers` | Plists layered under the request; the caller wins |
| `:rewrite-request`, `:rewrite-response` | Functions over the layered request / the result |

`make-provider` overrides `:base-url`, `:model` and `:api-key` per instance. A key comes from `:api-key` or the named environment variable and never appears in metadata. Without one, `complete` answers `(:bad-request ...)` and `describe-backend` reports `:status :unavailable`.

## Schemas

Tool parameters are typed schemas: `string`, `number`, `boolean`, `(integer lo hi)`, `(member :a :b)`, `(or null x)`, `(array-of x)`, `(map-of x)`, `(object ...)` and `any`. `schema->json-schema` renders one for a model, `json-schema->schema` reads one back, and `coerce-args` checks a model's arguments against one.

## Testing

`cl-inference/client/tests` runs offline against a fake HTTP server and passes on SBCL, CCL and ECL. See [Testing](testing.md).

## Limitations

| Missing | Tracker |
|---|---|
| In-process backend over sessions | [#29](https://todo.sr.ht/~takeiteasy/cl-inference/29) |
| One watcher thread per call | [#31](https://todo.sr.ht/~takeiteasy/cl-inference/31) |
| `json-get` is O(n²) on large objects | [#30](https://todo.sr.ht/~takeiteasy/cl-inference/30) |
| usocket connect-refusal workaround | [#32](https://todo.sr.ht/~takeiteasy/cl-inference/32) |
| Cancel-token actions are never removed | [#34](https://todo.sr.ht/~takeiteasy/cl-inference/34) |
| Socket leak window in `open-connection` | [#33](https://todo.sr.ht/~takeiteasy/cl-inference/33) |

[^retry]: `retry-after` is milliseconds, read from `retry-after-ms`, else `Retry-After` as seconds or an HTTP date.
[^sink]: Deltas are delivered inside the deadline, so a sink blocked on one is unwound when the deadline passes. The closing `:done` is delivered under its own bound, `*sink-grace*` seconds (default 5), so `complete` returns within `:timeout` plus `*sink-grace*`. The deadline is a watcher thread that shuts the socket down and interrupts the calling thread out of the exchange.
