# Architectures

`define-architecture` declares a model family. A child inherits from a parent and overrides only the entries that differ.

The library ships `local:llama` and `local:qwen3`. Qwen3 inherits Llama's blocks, metadata mappings and tied-output fallback, and declares only its differences:

```lisp
(define-architecture qwen3 (llama)
  (:blocks (attn attention :qk-norm t))
  (:format :gguf
    (:arch "qwen3")
    (:rope-pairing :neox)
    (:chat-template qwen3-chat)
    (:tensor attn.q-norm "blk.~d.attn_q_norm.weight")
    (:tensor attn.k-norm "blk.~d.attn_k_norm.weight")))
```

The definition above uses the library's package. Applications refer to the shipped classes as `local:llama` and `local:qwen3`, or declare their own child:

```lisp
(local:define-architecture llama-variant (local:llama)
  (:format :gguf (:arch "llama-variant")))
```

Loading picks the architecture the file declares:

```lisp
(local:with-weights (w "Qwen3-0.6B-Q8_0.gguf")
  (let ((arch (local:load-architecture w)))
    (local:hparam arch 'head-dim)                    ; 128
    (local:architecture-tensor-name arch 'attn.q 3)   ; "blk.3.attn_q.weight"
    (local:architecture-tensor arch w 'head.output))) ; output or tied embedding view
```

Qwen3 normalizes each Q/K head before NeoX RoPE. Its head dimension comes from GGUF metadata; for 0.6B it is 128, while `dim / n-heads` is 64. See [Qwen3](qwen3.md) for generation and full-model checks.

## Sections

| Section | Entry | Meaning |
|---|---|---|
| `(:hparams ...)` | `name` or `(name :default value)` | A hyperparameter, stored in a slot of the same name. No default makes it required |
| `(:blocks ...)` | `(slot class . initargs)` | A block that runs once per layer[^blocks] |
| `(:model-blocks ...)` | `(slot class . initargs)` | A block that runs once per model, such as the embedding and LM head |
| `(:format key ...)` | `(:arch "name")` | The `general.architecture` value that selects this architecture |
| | `(:hparam name "template")` | Metadata key for an hparam; `~a` is the `:arch` name |
| | `(:tensor role "template" [:optional t] [:fallback role])` | Tensor name for a role; `~d` is the layer index. `:fallback` reads another role when the file lacks this tensor |
| | `(:chat-template name)` | The [chat template](chat-templates.md) to use when the file carries no recognisable one |
| | `(:option value)` | Any other keyword, e.g. `:rope-pairing`; read with `architecture-format-option` |

Each format has its own section, so tensor names, metadata keys and RoPE pairing belong to the architecture and the format together.[^rope]

## Inheritance

Entries merge by symbol name, so a child in another package overrides what it inherits, along the class precedence list: hparam name, block slot, model-block slot, tensor role, format option, format hparam key. A child entry replaces the parent entry with the same name and leaves the rest alone. A child cannot remove an inherited entry ([limitation](limitations.md)). With several parents, the earlier parent wins.

Merging happens on lookup, so redefining a parent updates its children.

Because `~a` is the architecture's own `:arch`, `qwen3` reuses the `llama` key templates without repeating them.

[Tokenizers](tokenizers.md) and [chat templates](chat-templates.md) use the same inheritance rules.

## Functions

| Function | Returns |
|---|---|
| `load-architecture weights` | An instance of the most specific architecture whose `:arch` matches the file, with hparams filled from metadata |
| `hparam arch name` | An hparam value |
| `architecture-format arch` | The weights format the instance was loaded for |
| `architecture-tensor-name arch role &optional layer` | The file's tensor name |
| `architecture-tensor arch weights role &optional layer` | Tensor view, the `:fallback` role's view when absent, or NIL when `:optional` and absent |
| `architecture-format-option arch format option &optional default` | Option value and whether it was declared |
| `architecture-block-slots arch` / `architecture-block-spec arch slot` | Block slot names / `(class . initargs)` |
| `architecture-model-block-slots arch` / `architecture-model-block-spec arch slot` | The same for `:model-blocks` |
| `architecture-hparam-specs arch` | Alist of `(name . plist)` |

`arch` in the lookup functions is an instance, a class or a class name. Roles and hparams are matched by symbol name, so they need not be in the architecture's package.

## Errors

| Condition | Signalled when |
|---|---|
| `architecture-error` | Malformed definition, a required hparam missing from the file, an unknown hparam or tensor role, or several architectures matching one file |
| `unknown-architecture` | No architecture declares the file's architecture name for its format |

A non-optional tensor missing from the file signals `weights-error`.

Mods specialise methods on the architecture class, including with `:around`.

[^blocks]: Block slots hold a class name and initargs; `load-model` instantiates them. See [blocks](blocks.md).
[^rope]: GGUF `llama` permutes Q/K and uses adjacent-pair RoPE, HF safetensors uses rotate-half, and Qwen uses NeoX, so one architecture can need a different pairing per format.
