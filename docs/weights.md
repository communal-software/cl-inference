# Weights

Open a weight file in any registered format and read its tensors as cl-tensor views.

```lisp
(ci:with-weights (w "model.gguf")
  (list (ci:weights-metadata w "general.architecture")
        (ci:weights-tensor-names w)
        (ci:weights-tensor w "token_embd.weight")))
```

## Functions

| Function | Returns |
|---|---|
| `open-weights path &key format` | A `weights` object; `format` NIL detects it from the file header |
| `close-weights weights` | Nothing; repeat calls are harmless |
| `with-weights (var path &key format) body` | Runs `body`, then closes |
| `weights-format weights` | Format key, e.g. `:gguf` |
| `weights-metadata weights key &optional default` | Value (or `default`) and whether `key` was present |
| `weights-metadata-keys weights` | All metadata keys |
| `weights-tensor-names weights` | Tensor names in file order |
| `weights-tensor-info weights name` | `tensor-info`: name, shape, format type, byte count |
| `weights-tensor weights name` | Typed tensor view |
| `weights-tensor-bytes weights name` | Raw payload as a `:u8` tensor view |

Shapes are row-major: the last axis is contiguous, whatever order the file format stores.

## Lifetime

Tensor views point into the file mapping; nothing is copied.[^views] Views are invalid after `close-weights`. Calling an accessor on closed weights signals `weights-closed`.

## Errors

| Condition | Signalled when |
|---|---|
| `weights-error` | Base condition; also an unknown tensor name |
| `unknown-weights-format` | No registered format matches, or `format` is not registered |
| `weights-closed` | Accessor used after `close-weights` |
| `unsupported-weights-type` | A tensor's storage type has no typed view; `weights-tensor-bytes` still works |

## Add a format

Subclass `weights`, implement the generics, and register the format.

| Generic | Contract |
|---|---|
| `weights-metadata-ref` | Value and presence for a key |
| `weights-metadata-key-list` | Every key |
| `weights-tensor-info-list` | Every `tensor-info`, in file order |
| `weights-tensor-info-ref` | One `tensor-info`, or NIL |
| `weights-make-tensor` | Typed view, or signal `unsupported-weights-type` |
| `weights-make-bytes` | `:u8` view of the raw payload |
| `weights-architecture-name` | Architecture name the file declares, or NIL (the default); used by [`load-architecture`](architectures.md) |
| `weights-release` | Free resources; called once |

```lisp
(ci:register-weights-format :my-format
  :detect (lambda (path header-octets) ...)
  :open   (lambda (path) (make-instance 'my-weights :format :my-format :path path)))
```

The first 16 bytes of the file are passed to `:detect`. Formats are tried in registration order. See [GGUF](gguf.md) for the built-in format.

[^views]: Views are ordinary tensors over storage the caller owns, so cl-tensor's [foreign storage rules](https://git.sr.ht/~takeiteasy/cl-tensor/tree/trunk/item/docs/design.md) apply.
