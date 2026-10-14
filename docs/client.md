# Client

`cl-inference` is one synchronous `complete` contract for chat models, with HTTP backends and the [in-process `:local` backend](local-backend.md) behind it. The system loads no cl-tensor, so HTTP-only consumers do not load the engine.

```lisp
(asdf:load-system :cl-inference)
(use-package :cl-inference)

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
| `:temperature`, `:top-p`, `:top-k`, `:max-tokens`, `:stop`, `:seed` | Sampling; each protocol advertises the ones it renders |
| `:timeout` | Milliseconds for the whole exchange. Default 30000 |
| `:cancel` | A token from `make-cancel-token`; `(cancel token)` ends the call |
| `:stream` | A function called with each [event](#events) |
| `:ref` | Stamped on every event |
| `:include-usage` | OpenAI-compatible streams: T (default) requests token usage; NIL omits the streaming option |

`:content` is a string or a list of blocks, `(:type :text :text "...")`.

## Reply

```lisp
(:role :assistant :content <blocks or nil> :tool-calls <calls> :done t
 :meta (:finish-reason :stop :usage (:prompt-tokens 7 ...) :id "..."))
```

OpenAI-compatible streams request usage and collect the trailing usage chunk into `:meta :usage`. A server that omits usage leaves it absent. The reader waits through `[DONE]` or EOF after the finish reason, within the request deadline.[^stream-usage]

For a compatible server that rejects `stream_options`, pass `:include-usage nil`:

```lisp
(complete :protocol-openai :base-url "http://127.0.0.1:8080/v1"
          :model "local-model" :messages '((:role :user :content "Hello"))
          :include-usage nil :stream #'print)
```

A provider may set `:defaults '(:include-usage nil)`; an explicit request value wins. With NIL, the reader returns at the finish reason. Non-streamed requests omit `stream_options`.

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
| `local-backend` | `:local` | A model in this image; configure with [`make-local-backend`](local-backend.md) |

`describe-backend` returns a metadata plist (`:kind`, `:name`, `:summary`, ...). `(backends :kind :provider)` lists registered names.

Writing a backend means subclassing `backend` and adding a method:

```lisp
(defmethod backend-complete ((backend my-backend) request)
  (list :ok (list :role :assistant :content (normalize-content "hi") :done t)))
```

`request` is already checked. The method honours `:cancel`, `:timeout` and `:stream`.

## Providers

A provider is data layered under each request: a protocol, base URL, authentication, model catalogue and quirks.

| Built-in | Base URL | Authentication |
|---|---|---|
| `:openai` | `https://api.openai.com/v1` | Bearer key from `OPENAI_API_KEY` or `:api-key` |
| `:ollama` | `http://127.0.0.1:11434` | None |

```lisp
(complete :openai :model "your-model-id"
                 :messages '((:role :user :content "Reply with ok.")))
```

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

`validate-schema` signals a definition error for malformed schemas. A conditional
requirement uses `:required-when (:controller :value)` or
`:required-when (:controller (:value ...))`; the controller is another member
parameter in the same schema.

## Helpers

For consumers that layer their own services over the client.

`on-cancel` returns an idempotent, zero-argument remover. A cancelled token
calls the action immediately. Removal prevents actions still registered on the
token; an action already claimed by `cancel` may run. Completions remove their
actions when they finish, so a token can be reused across calls until cancelled.

```lisp
(let ((remove-action (on-cancel token #'stop-work)))
  (unwind-protect (do-work)
    (funcall remove-action)))
```

| Symbol | Does |
|---|---|
| `tool-schema` | A tool metadata plist's `:params` |
| `param-name`, `param-type`, `param-options` | Read one `(:name type . options)` schema entry |
| `*absent*` | Marks a parameter the caller did not supply |
| `json-object`, `json-get` | Build and read a JSON object; `(setf json-get)` sets a key |
| `json-object-alist` | A JSON object's `(key . value)` pairs in insertion order |
| `tools->json`, `call-arguments->json`, `wire-tool-name` | The wire form of a tool list, a call's arguments and a tool's name |
| `untyped->json`, `json->arguments` | Render and read an argument plist without a schema |
| `lisp-tool-name` | A wire tool name as a keyword |
| `reply-prompt-tokens`, `done-reason` | Read a reply's prompt size and a result's finish reason |
| `call-with-completion-deadline` | Run a zero-argument result function under a request's deadline, with one bounded closing event |
| `call-with-deadline` | Run a function on this thread, bounded by milliseconds and a `:cancel` token |
| `header-alist` | A header plist as a lower-cased alist |

## Testing

`cl-inference/tests` runs offline against a fake HTTP server and passes on SBCL, CCL and ECL. See [Testing](testing.md).

## Limitations

| Missing | Tracker |
|---|---|
| One watcher thread per call | [#26](https://github.com/communal-software/cl-inference/issues/26) |
| `json-get` is O(n²) on large objects | [#25](https://github.com/communal-software/cl-inference/issues/25) |
| usocket connect-refusal workaround | [#27](https://github.com/communal-software/cl-inference/issues/27) |

[^retry]: `retry-after` is milliseconds, read from `retry-after-ms`, else `Retry-After` as seconds or an HTTP date.
[^sink]: Deltas are delivered inside the deadline, so a sink blocked on one is unwound when the deadline passes. The closing `:done` is delivered under its own bound, `*sink-grace*` seconds (default 5). During socket acquisition, cancellation and deadline unwinding wait until the socket is recorded for cleanup or connect fails; the connect timeout is `:timeout` rounded up to whole seconds, with a one-second minimum. The deadline is a watcher thread that shuts the socket down and interrupts the calling thread out of the exchange.

[^stream-usage]: A stalled trailer times out even after a finish reason; cancellation also remains active. No usage is invented and rejected requests are not retried. The [OpenAI API reference](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create) describes the empty-choices usage chunk before `[DONE]`.
