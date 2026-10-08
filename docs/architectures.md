# Architectures

`define-architecture` declares a model family. A child inherits from a parent and overrides only the entries that differ.

```lisp
(ci:define-architecture llama ()
  (:hparams (n-layers) (dim) (rope-theta :default 10000.0))
  (:blocks (attn attention) (mlp swiglu-mlp))
  (:format :gguf
    (:arch "llama")
    (:rope-pairing :adjacent)
    (:hparam n-layers "~a.block_count")
    (:hparam dim "~a.embedding_length")
    (:hparam rope-theta "~a.rope.freq_base")
    (:tensor attn.q "blk.~d.attn_q.weight")
    (:tensor output "output.weight" :optional t)))

(ci:define-architecture qwen3 (llama)
  (:blocks (attn attention :qk-norm rmsnorm))
  (:format :gguf
    (:arch "qwen3")
    (:rope-pairing :neox)
    (:tensor attn.q-norm "blk.~d.attn_q_norm.weight")))
```

Loading picks the architecture the file declares:

```lisp
(ci:with-weights (w "qwen3.gguf")
  (let ((arch (ci:load-architecture w)))      ; => #<QWEN3>
    (ci:hparam arch 'n-layers)                ; from "qwen3.block_count"
    (ci:architecture-tensor-name arch 'attn.q 3)   ; "blk.3.attn_q.weight"
    (ci:architecture-tensor arch w 'output)))      ; a view, or NIL when absent
```

## Sections

| Section | Entry | Meaning |
|---|---|---|
| `(:hparams ...)` | `name` or `(name :default value)` | A hyperparameter, stored in a slot of the same name. No default makes it required |
| `(:blocks ...)` | `(slot class . initargs)` | A block slot, stored as data[^blocks] |
| `(:format key ...)` | `(:arch "name")` | The `general.architecture` value that selects this architecture |
| | `(:hparam name "template")` | Metadata key for an hparam; `~a` is the `:arch` name |
| | `(:tensor role "template" [:optional t])` | Tensor name for a role; `~d` is the layer index |
| | `(:option value)` | Any other keyword, e.g. `:rope-pairing`; read with `architecture-format-option` |

Each format has its own section, so tensor names, metadata keys and RoPE pairing belong to the architecture and the format together.[^rope]

## Inheritance

Entries merge by name along the class precedence list: hparam name, block slot, tensor role, format option, format hparam key. A child entry replaces the parent entry with the same name and leaves the rest alone. A child cannot remove an inherited entry ([limitation](limitations.md)). With several parents, the earlier parent wins.

Merging happens on lookup, so redefining a parent updates its children.

Because `~a` is the architecture's own `:arch`, `qwen3` reuses the `llama` key templates without repeating them.

## Functions

| Function | Returns |
|---|---|
| `load-architecture weights` | An instance of the most specific architecture whose `:arch` matches the file, with hparams filled from metadata |
| `hparam arch name` | An hparam value |
| `architecture-format arch` | The weights format the instance was loaded for |
| `architecture-tensor-name arch role &optional layer` | The file's tensor name |
| `architecture-tensor arch weights role &optional layer` | Tensor view; NIL when `:optional` and absent |
| `architecture-format-option arch format option &optional default` | Option value and whether it was declared |
| `architecture-block-slots arch` / `architecture-block-spec arch slot` | Block slot names / `(class . initargs)` |
| `architecture-hparam-specs arch` | Alist of `(name . plist)` |

`arch` in the lookup functions is an instance, a class or a class name.

## Errors

| Condition | Signalled when |
|---|---|
| `architecture-error` | Malformed definition, a required hparam missing from the file, an unknown hparam or tensor role, or several architectures matching one file |
| `unknown-architecture` | No architecture declares the file's architecture name for its format |

A non-optional tensor missing from the file signals `weights-error`.

Mods specialise methods on the architecture class, including with `:around`.

[^blocks]: Block slots hold a class name and initargs; nothing is instantiated. Blocks are listed under [Limitations](limitations.md).
[^rope]: GGUF `llama` permutes Q/K and uses adjacent-pair RoPE, HF safetensors uses rotate-half, and Qwen uses NeoX, so one architecture can need a different pairing per format.
