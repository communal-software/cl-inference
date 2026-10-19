# Tokenizers

`load-tokenizer` reads the tokenizer a weight file carries and turns text into token ids and back. Tokenizers are declared with `define-tokenizer`, the same inheritance DSL as [architectures](architectures.md).

```lisp
(local:with-weights (w "Llama-3.2-1B-Instruct-f16.gguf")
  (let ((tokenizer (local:load-tokenizer w)))     ; => #<LLAMA-BPE>
    (let ((ids (local:encode tokenizer "Hello, world!")))   ; begins with the BOS id
      (local:decode tokenizer ids))))                      ; => "Hello, world!"
```

The tokenizer keeps its own copy of the vocabulary, so it stays valid after `w` closes.

## Functions

| Function | Returns |
|---|---|
| `load-tokenizer weights` | The declared tokenizer; the default for GGUF `gpt2` with missing or empty `pre` |
| `encode tokenizer text &key special add-bos` | Token ids, a `(simple-array (unsigned-byte 32) (*))` |
| `decode tokenizer ids &key special` | Text; invalid UTF-8 becomes U+FFFD |
| `make-detokenizer tokenizer &key special` | A [streaming decoder](#streaming) |
| `pre-tokenize tokenizer text` | The pieces a split-based BPE tokenizer produces; SentencePiece signals `tokenizer-error` |
| `token-id tokenizer string` | Id of a vocabulary entry, or NIL |
| `token-bytes tokenizer id` | The bytes a token stands for |
| `tokenizer-vocab-size tokenizer` | Number of tokens |
| `tokenizer-special-id tokenizer key` | Id the file declares for `:bos`, `:eos`, `:eot`, `:eom`, `:pad`, `:unk` or `:sep`, or NIL |
| `tokenizer-special-p tokenizer id` | True for control, unknown and user-defined tokens |
| `tokenizer-add-bos-p tokenizer` | The file's `add_bos_token`, or the tokenizer's default |
| `tokenizer-ignore-merges-p tokenizer` | Whether whole pieces found in the vocabulary skip merging |

Empty text follows the same BOS policy: `:add-bos nil` returns an empty unsigned-32 vector; `:add-bos t` returns one BOS ID or signals `tokenizer-error` if none is configured.

```lisp
(local:encode tokenizer "" :add-bos nil) ; => #(), unsigned-32 elements
```

`add-bos` defaults to the file's `add_bos_token`. Stop tokens are not decided here; read them with `tokenizer-special-id`, or `chat-template-stop-ids` from a [chat template](chat-templates.md).

## Special tokens

| Token type | `encode` matches it in text | `decode` renders it |
|---|---|---|
| Normal | n/a | always |
| User-defined, e.g. `<think>` | always | always |
| Control, e.g. `<\|eot_id\|>` | only with `:special t` | only with `:special t` |
| Unknown | only with `:special t` | only with `:special t` |
| Unused | never | never |

Without `:special t`, control token text such as `<s>` encodes as ordinary characters.

## Streaming

`detokenizer-push` returns the text a token completes. It returns `""` while a character is only partly received, so a multi-byte character split across tokens never appears broken.

```lisp
(let ((d (local:make-detokenizer tokenizer)))
  (loop for id across (local:encode tokenizer "Hi 😀" :add-bos nil)
        collect (local:detokenizer-push d id)))   ; "" for each token that ends mid-character
```

`detokenizer-finish` returns U+FFFD for bytes left dangling.

[HF SentencePiece BPE](hf-sentencepiece.md#streaming) buffers complete byte-token runs and releases their text at a run boundary or finish. Always append the text returned by `detokenizer-finish`.

## Declaring a tokenizer

```lisp
(local:define-tokenizer llama-bpe (local:byte-level-bpe)
  (:pre-tokenizer
    (contractions :case-insensitive t)
    (letters :prefix :non-letter-non-newline)
    (digits :max 3)
    (punctuation :leading-space t :trailing-newlines t)
    (newlines)
    (whitespace))
  (:format :gguf
    (:model "gpt2")
    (:pre "llama-bpe" "llama3" "llama-v3")))

(local:define-tokenizer qwen2 (llama-bpe)
  (:pre-tokenizer (digits :max 1))
  (:format :gguf (:pre "qwen2")))
```

| Section | Entry | Meaning |
|---|---|---|
| `(:pre-tokenizer ...)` | `(kind . options)` | A rule of the split that precedes merging. Rules are tried in order at each position[^regex] |
| `(:format key ...)` | `(:model "name")` | The `tokenizer.ggml.model` value that selects this tokenizer |
| | `(:pre "name" ...)` | The `tokenizer.ggml.pre` values that select it |
| | `(:ignore-merges bool)` | Emit a piece found whole in the vocabulary as one token, skipping merges[^merges] |
| | `(:add-bos bool)` | Whether to add BOS when the file has no `add_bos_token` |

Loading picks the most specific tokenizer whose `:model` equals the file's and whose `:pre` contains the file's. For GGUF model `gpt2`, missing or empty `tokenizer.ggml.pre` selects `default-bpe`, as does explicit `"default"`. GGUF model `llama` selects `llama-spm` independently of `pre`; see [SentencePiece](sentencepiece.md). Unknown BPE pre-types and other model types signal `unknown-tokenizer`.

[`:pre-tokenizer-stages`](pre-tokenizer-stages.md) declares sequential splitting passes. Flat rules and stages cannot be mixed in one inheritance lineage.

### Rules

| Rule | Options | Matches |
|---|---|---|
| `contractions` | `:case-insensitive` | `'s 't 'm 'd 're 've 'll` |
| `letters` | `:prefix :space` or `:non-letter-non-newline`, `:include-marks` | Letters and optional Unicode marks, optionally prefixed by a literal space or one character outside letters, numbers, CR and LF |
| `digits` | `:min n`, `:max n`, `:prefix :space`, `:ascii-only t` | A number run; minimum and maximum default to 1, NIL maximum is unbounded |
| `punctuation-symbols` | none | Unicode punctuation and `$+<=>^~\|` |
| `punctuation` | `:leading-space`, `:trailing-newlines`, `:exclude-marks` | A run outside whitespace, letters and numbers, optionally excluding Unicode marks, with an optional leading space and trailing newlines |
| `newlines` | none | Whitespace up to its last newline |
| `whitespace` | none | A whitespace run; before text it leaves its last character for the next piece |

Letters, marks, numbers and whitespace follow Unicode categories.[^unicode] The mark options default to NIL. Rule options are checked when the tokenizer loads, so a misspelled option signals `tokenizer-error`. A mod adds a rule kind by specializing `rule-matcher`:

```lisp
(defmethod local:rule-matcher ((kind (eql :hash)) options)
  (lambda (text start) (and (char= (schar text start) #\#) (1+ start))))
```

### Inheritance

Rules merge by kind along the class precedence list. Without placement, a child's rule replaces the parent's in place; a new kind appends. `:before kind` or `:after kind` inserts a new rule or moves an override beside an existing rule. Names match across packages.

```lisp
(local:define-tokenizer single-letter-bpe (local:llama-bpe)
  (:pre-tokenizer (single-letter :before letters))
  (:format :gguf (:pre "single-letter")))

(defmethod local:rule-matcher ((kind (eql :single-letter)) options)
  (declare (ignore options))
  (lambda (text start)
    (and (alpha-char-p (char text start)) (1+ start))))
```

With a matching vocabulary, `pre-tokenize` of `"hello"` returns `("h" "e" "l" "l" "o")`.

Entries resolve in declaration order. The anchor must already exist in the accumulated rules; forward references signal `tokenizer-error`. Each insertion goes immediately beside its anchor: after `(a b)`, adding `x :after a` then `y :after a` produces `(a y x b)`. Repeated `:before` insertions retain declaration order.

Placement accepts one non-NIL symbol anchor. Self-references, repeated placement keys and combining `:before` with `:after` signal `tokenizer-error` during macro expansion. Missing anchors fail when effective rules are queried or the tokenizer loads. Placement keys are absent from effective rule options and never reach `rule-matcher`.

`:model` and `:pre` merge by name, so a child that lists `:pre` replaces the parent's names. Redefining a parent updates its children; loaded tokenizers keep their compiled matchers.

### Standard tokenizers

| Tokenizer | `tokenizer.ggml.pre` | Digits | `:ignore-merges` | `:add-bos` |
|---|---|---|---|---|
| `default-bpe` | missing, empty, `default` | staged Unicode runs and ASCII triples | no | no |
| `llama-bpe` | `llama-bpe`, `llama3`, `llama-v3`, `falcon3`, `falcon-h1`, `pixtral`, `midm-2.0`, `lfm2` | up to 3 | yes | yes |
| `qwen2` | `qwen2`, `deepseek-r1-qwen`, `kormo` | 1 | no | no |
| `qwen35` | `qwen35` | 1 | no | no |

All four use `tokenizer.ggml.model` `gpt2`. `qwen2` inherits `llama-bpe`, so it sets both options itself. `default-bpe` has its own [four-stage split](pre-tokenizer-stages.md#default-gguf-pipeline); it is not an alias for plain GPT-2 splitting.

## Qwen3.5 GGUF

`qwen35` inherits Qwen2’s rule order and includes Unicode marks in letter runs. For example, `pre-tokenize` of decomposed `"école"` returns `("école")`. Standalone marks also form pieces. Punctuation runs exclude marks; number pieces contain one Unicode number each. Input and decoded bytes preserve the original normalization form.[^qwen35] GGUF `qwen35` treats a vocabulary entry named `</s>` as a control token even when its declared type is normal; it matches and renders only with `:special t`.

This coverage applies to GGUF tokenization. Qwen3.5 inference requires its own architecture support; HF tokenizer coverage remains listed [separately](hf-tokenizers.md).

## SentencePiece

`llama-spm` reads GGUF Llama SentencePiece vocabularies and merges by token score. Decoding preserves token-piece spaces, including the synthetic leading space. See [SentencePiece behavior and examples](sentencepiece.md).

[Gemma 3 270M IT](gemma.md) uses the same `llama-spm` implementation with prefix spaces disabled and model-specific BOS/EOS IDs. Its pinned vocabulary has separate reference parity coverage.

## HF files

The optional Safetensors system reads the pinned Llama, Qwen3, TinyLlama and Gemma `tokenizer.json` pipelines and special-token configuration. See [HF tokenizers](hf-tokenizers.md) for input validation and coverage.

## Errors

| Condition | Signalled when |
|---|---|
| `tokenizer-error` | Malformed definition or vocabulary, an invalid token id, `add-bos` with no BOS token, or several tokenizers matching one file |
| `unknown-tokenizer` | No tokenizer matches the file's model and `pre`; `unknown-tokenizer-model` and `unknown-tokenizer-pre` return them |

Byte-level BPE loading signals `tokenizer-error` for a vocabulary that is not byte-level text, a merge that is not `"left right"` of known tokens, a missing byte token, a `token_type` list of the wrong length, and a special-token id outside the vocabulary.

## Limitations

| Missing | Tracker |
|---|---|
| Other GGUF BPE `pre` types | [#61](https://github.com/communal-software/cl-inference/issues/61) |
| GGUF SentencePiece variants outside pinned Llama/Gemma coverage | [#62](https://github.com/communal-software/cl-inference/issues/62) |
| Automatic EOS insertion and extra-whitespace normalization | [#65](https://github.com/communal-software/cl-inference/issues/65), [#66](https://github.com/communal-software/cl-inference/issues/66) |
| Chat templates | [chat templates](chat-templates.md#limitations) |

[^regex]: The flat standard rules reproduce llama.cpp's `llama3` and `qwen2` patterns, for example `\p{N}{1,3}` and `\s+(?!\S)`, as hand-written scanners. The tests compare both against the patterns run through Python's `regex` module.
[^merges]: The llama 3 merge list does not reach every token, so merging alone encodes some words differently from llama.cpp.
[^unicode]: Letters, marks and numbers are the general categories `L*`, `M*` (`Mn`, `Mc`, `Me`) and `N*`; whitespace is the `White_Space` property. A cache covers the Basic Multilingual Plane.

[^qwen35]: Split, token-ID and decoded-byte expectations use llama.cpp revision `8bb3cf0d216a5ae48360179f2931daf284be4d01` and its checksum-pinned `ggml-vocab-qwen35.gguf`. See [reference generation](testing.md#qwen35-reference).
