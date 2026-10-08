# Testing

The suite uses FiveAM.

```sh
sbcl --non-interactive --eval '(require :asdf)' --eval '(asdf:test-system :cl-inference)'
```

`asdf:test-system` signals an error if any check fails. From a REPL, `(cl-inference/tests:run-tests)` returns true when every check passes.

| File | Covers |
|---|---|
| `tests/systems.lisp` | Every subsystem package loads |
| `tests/gguf-writer.lisp` | Test-only GGUF writer with knobs for malformed files |
| `tests/gguf.lisp` | GGUF metadata, tensor views, half precision, copy-on-write and every rejection case |
| `tests/weights.lisp` | Weights protocol through a mock format: detection, metadata, tensors, closing |
| `tests/architecture.lisp` | Architecture DSL: inheritance, hparams, tensor names, loading from GGUF, malformed definitions |

Full-model checks compare against a dev-only reference kept in the gitignored `reference/` directory.
