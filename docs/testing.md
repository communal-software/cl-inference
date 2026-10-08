# Testing

The suite uses FiveAM.

```sh
sbcl --non-interactive --eval '(require :asdf)' --eval '(asdf:test-system :cl-inference)'
```

`asdf:test-system` signals an error if any check fails. From a REPL, `(cl-inference/tests:run-tests)` returns true when every check passes.

| File | Covers |
|---|---|
| `tests/systems.lisp` | Every subsystem package loads |

Full-model checks compare against a dev-only reference kept in the gitignored `reference/` directory.
