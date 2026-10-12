# Testing

The suite uses FiveAM.

```sh
sbcl --non-interactive --eval '(require :asdf)' --eval '(asdf:test-system :cl-inference)'        # client
sbcl --non-interactive --eval '(require :asdf)' --eval '(asdf:test-system :cl-inference/local)'  # engine
```

`asdf:test-system` signals an error if any check fails. From a REPL, `(cl-inference/tests:run-tests)` and `(cl-inference/local/tests:run-tests)` return true when every check passes.

The [client](client.md) suite loads neither the engine nor cl-tensor and passes on SBCL, CCL and ECL.

| File | Covers |
|---|---|
| `tests/local/systems.lisp` | Every subsystem package loads; the `:local` backend is registered |
| `tests/local/gguf-writer.lisp` | Test-only GGUF writer with knobs for malformed files |
| `tests/local/gguf.lisp` | GGUF metadata, tensor views, half precision, copy-on-write and every rejection case |
| `tests/local/weights.lisp` | Weights protocol through a mock format: detection, metadata, tensors, closing |
| `tests/local/architecture.lisp` | Architecture DSL: inheritance, model blocks, tensor fallbacks, hparams, tensor names, loading from GGUF, malformed definitions |
| `tests/local/tokenizer.lisp` | Tokenizer DSL, both pre-tokenizers against llama.cpp's patterns, BPE merging, special tokens, streaming decode, malformed vocabularies |
| `tests/local/blocks.lisp` | Blocks and `model-forward` against a double-precision reference: both RoPE pairings, frequency factors, partial RoPE, GQA, QK-norm, tied head, cache |

| File | Covers |
|---|---|
| `tests/schema.lisp` | Parameter schemas: coercion, JSON Schema in both directions |
| `tests/transport.lisp` | Retry-After, cancel tokens, the exchange deadline |
| `tests/openai.lisp`, `ollama.lisp` | Wire request and reply, streaming, deadlines and cancellation against a fake HTTP server |
| `tests/provider.lisp` | Declarations, credentials, layering, quirks |
| `tests/scripted.lisp` | `complete` / `complete*` contract and the scripted backend |
| `tests/systems.lisp` | The client depends on neither the engine nor cl-tensor |
| `tests/fake-http.lisp` | Offline HTTP server used by the above |

Live Ollama tests are skipped unless `CL_INFERENCE_OLLAMA_URL` (the `/v1` route) or `CL_INFERENCE_OLLAMA_NATIVE_URL` is set; `CL_INFERENCE_OLLAMA_MODEL` picks the model.

Full-model checks compare against a dev-only reference kept in the gitignored `reference/` directory, which also holds llama.cpp's `ggml-vocab-llama-bpe` and `ggml-vocab-qwen2` fixtures for the tokenizer.
