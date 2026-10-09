# Testing

The suite uses FiveAM.

```sh
sbcl --non-interactive --eval '(require :asdf)' --eval '(asdf:test-system :cl-inference)'
```

`asdf:test-system` signals an error if any check fails. From a REPL, `(cl-inference/tests:run-tests)` returns true when every check passes. It runs the engine suite, then the client suite.

The [client](client.md) suite loads neither the engine nor cl-tensor and passes on SBCL, CCL and ECL:

```sh
sbcl --non-interactive --eval '(require :asdf)' --eval '(asdf:test-system :cl-inference/client)'
```

| File | Covers |
|---|---|
| `tests/systems.lisp` | Every subsystem package loads |
| `tests/gguf-writer.lisp` | Test-only GGUF writer with knobs for malformed files |
| `tests/gguf.lisp` | GGUF metadata, tensor views, half precision, copy-on-write and every rejection case |
| `tests/weights.lisp` | Weights protocol through a mock format: detection, metadata, tensors, closing |
| `tests/architecture.lisp` | Architecture DSL: inheritance, hparams, tensor names, loading from GGUF, malformed definitions |

| File | Covers |
|---|---|
| `tests/client/schema.lisp` | Parameter schemas: coercion, JSON Schema in both directions |
| `tests/client/transport.lisp` | Retry-After, cancel tokens, the exchange deadline |
| `tests/client/openai.lisp`, `ollama.lisp` | Wire request and reply, streaming, deadlines and cancellation against a fake HTTP server |
| `tests/client/provider.lisp` | Declarations, credentials, layering, quirks |
| `tests/client/scripted.lisp` | `complete` / `complete*` contract and the scripted backend |
| `tests/client/systems.lisp` | The client depends on neither the engine nor cl-tensor |
| `tests/client/fake-http.lisp` | Offline HTTP server used by the above |

Live Ollama tests are skipped unless `CL_INFERENCE_OLLAMA_URL` (the `/v1` route) or `CL_INFERENCE_OLLAMA_NATIVE_URL` is set; `CL_INFERENCE_OLLAMA_MODEL` picks the model.

Full-model checks compare against a dev-only reference kept in the gitignored `reference/` directory.
