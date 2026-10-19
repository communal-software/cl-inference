# Architectures

`define-architecture` declares a model family. A child inherits from a parent and overrides only the entries that differ.

The [support overview](support.md#local-architectures) lists the shipped and verified models. The library ships `local:llama`, `local:qwen3` and [CPU GGUF `local:gemma3`](gemma.md). Qwen3 inherits Llama's blocks, metadata mappings and tied-output fallback, and declares only its differences:

```lisp
(define-architecture qwen3 (llama)
  (:blocks (attn attention :qk-norm t))
  (:format :gguf
    (:arch "qwen3")
    (:rope-pairing :neox)
    (:chat-template qwen3-chat)
    (:tensor attn.q-norm "blk.~d.attn_q_norm.weight")
    (:tensor attn.k-norm "blk.~d.attn_k_norm.weight"))
  (:format :safetensors
    (:arch "qwen3")
    (:tensor attn.q-norm "model.layers.~d.self_attn.q_norm.weight")
    (:tensor attn.k-norm "model.layers.~d.self_attn.k_norm.weight")))
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

Qwen3 normalizes each Q/K head before NeoX RoPE. Its head dimension comes from GGUF metadata; for 0.6B it is 128, while `dim / n-heads` is 64. See [Sessions](sessions.md) for generation and [Model acceptance](acceptance.md) for full-model checks.

## Sections

| Section | Entry | Meaning |
|---|---|---|
| `(:hparams ...)` | `name` or `(name :default value)` | A hyperparameter, stored in a slot of the same name. No default makes it required |
| `(:blocks ...)` | `(slot class . initargs)` | A block that runs once per layer[^blocks] |
| `(:model-blocks ...)` | `(slot class . initargs)` | A block that runs once per model, such as the embedding and LM head |
| `(:format key ...)` | `(:arch "name")` | The name returned by `weights-architecture-name`: GGUF `general.architecture` or HF `model_type` |
| | `(:hparam name "template")` | Metadata key for an hparam; `~a` is the `:arch` name |
| | `(:tensor role "template" [:optional t] [:fallback role])` | Tensor name for a role; `~d` is the layer index. `:fallback` reads another role when the file lacks this tensor |
| | `(:tensor role nil :optional t)` | This format has no stored tensor for the role; tensor lookup returns NIL |
| | `(:chat-template name)` | The [chat template](chat-templates.md) to use when the file carries no recognisable one |
| | `(:option value)` | Any other keyword, e.g. `:rope-pairing`; read with `architecture-format-option` |

Each format has its own section, so tensor names, metadata keys and RoPE pairing belong to the architecture and the format together.[^rope]

Llama, Qwen3 and Gemma 3 provide `:gguf` and `:safetensors` mappings. Gemma HF uses `model_type="gemma3_text"`; its [format-specific norms and attention](gemma-hf.md#model-behavior) follow the pinned text configuration. HF values keep their configuration names, including `hidden_size`, `num_hidden_layers`, `head_dim` and `rope_scaling`; JSON null becomes NIL in hparams. See [Safetensors](safetensors.md#model-loading) for CPU model coverage and configuration restrictions.

## Inheritance

Entries merge by symbol name, so a child in another package overrides what it inherits, along the class precedence list: hparam name, block slot, model-block slot, tensor role, format option, format hparam key. A child entry replaces the parent entry with the same name and leaves the rest alone. With several parents, the earlier parent wins.

Remove inherited entries with `(:remove name)` in `:hparams`, `:blocks` or `:model-blocks`. Inside a format, name the kind:

```lisp
(local:define-architecture trimmed-model (local:llama)
  (:hparams (:remove rope-theta))
  (:blocks (:remove mlp))
  (:format :gguf
    (:remove :tensor head.output)
    (:remove :hparam head-dim)
    (:remove :option :rope-pairing)))
```

Declarations and removals follow the same precedence and declaration order. Removing an absent name is a no-op; a later declaration restores it. Removed tensor roles are unknown; removed options return the caller's default and a false presence value.

Removing an hparam excludes it from loading and `hparam` lookup, and hides its metadata mappings in every format. Restoring it restores inherited mappings unless those mappings are explicitly removed.[^removed-slots]

Merging happens on lookup, so redefining a parent updates its children.

A child inherits selectors for every format. `(:format :safetensors (:arch nil))` disables automatic Safetensors selection for that child while retaining the inherited mappings.

Because `~a` is the architecture's own `:arch`, `qwen3` reuses the `llama` key templates without repeating them.

[Tokenizers](tokenizers.md) and [chat templates](chat-templates.md) use the same inheritance rules.

## Functions

| Function | Returns |
|---|---|
| `load-architecture weights` | An instance of the most specific architecture whose `:arch` matches the file, with hparams filled from metadata |
| `hparam arch name` | An hparam value |
| `architecture-format arch` | The weights format the instance was loaded for |
| `architecture-tensor-name arch role &optional layer` | The file's tensor name, or NIL for an explicitly unstored optional role |
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

## Extend from an application package

An application uses exported classes and defines its own block subclass. This observer records attention output shapes and returns the original delta:

```lisp
(defpackage #:my-model
  (:use #:cl)
  (:local-nicknames (#:local #:cl-inference/local) (#:ct #:cl-tensor)))
(in-package #:my-model)

(defvar *observations* nil)
(defclass observing-attention (local:attention) ())

(defmethod local:block-forward :around
    ((block observing-attention) x &key &allow-other-keys)
  (let ((delta (call-next-method)))
    (push (list :layer (local:block-layer block)
                :shape (coerce (ct:tensor-shape delta) 'list))
          *observations*)
    delta))

(local:define-architecture observed-llama (local:llama)
  (:blocks (attn observing-attention))
  (:format :gguf (:arch "observed-llama"))
  (:format :safetensors (:arch nil)))
```

The child inherits hparams, tensor roles and the MLP. Its `attn` entry replaces the inherited entry even though the symbols belong to different packages.

The unique GGUF selector matches an `observed-llama` fixture; inherited metadata templates read `observed-llama.embedding_length`, for example. Ordinary Llama files keep their existing selector. A child that retains `:arch "llama"` becomes the most specific selector for those files; unrelated matching children cause an ambiguity error. NIL disables the inherited Safetensors selector.

[Runnable examples](examples.md#run-without-downloads) exercise this definition against a tiny fixture and verify unchanged logits. Specialize `layer-forward` on the child architecture when the modification changes how block outputs combine.

[^blocks]: Block slots hold a class name and initargs; `load-model` instantiates them. See [blocks](blocks.md).
[^rope]: GGUF `llama` permutes Q/K and uses adjacent-pair RoPE, HF safetensors uses rotate-half, and Qwen uses NeoX, so one architecture can need a different pairing per format.

[^removed-slots]: CLOS retains inherited slots internally; their presence does not make a removed hparam public.
