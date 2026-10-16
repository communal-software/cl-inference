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
| `pre-tokenize tokenizer text` | The pieces `text` splits into before merging |
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

Loading picks the most specific tokenizer whose `:model` equals the file's and whose `:pre` contains the file's. For GGUF model `gpt2`, missing or empty `tokenizer.ggml.pre` selects `default-bpe`, as does explicit `"default"`. Unknown names and other model types still signal `unknown-tokenizer`.

[`:pre-tokenizer-stages`](pre-tokenizer-stages.md) declares sequential splitting passes. Flat rules and stages cannot be mixed in one inheritance lineage.

### Rules

| Rule | Options | Matches |
|---|---|---|
| `contractions` | `:case-insensitive` | `'s 't 'm 'd 're 've 'll` |
| `letters` | `:prefix :space` or `:non-letter-non-newline` | Letters, optionally prefixed by a literal space or one non-letter, non-digit, non-newline character |
| `digits` | `:min n`, `:max n`, `:prefix :space`, `:ascii-only t` | A number run; minimum and maximum default to 1, NIL maximum is unbounded |
| `punctuation-symbols` | none | Unicode punctuation and `$+<=>^~\|` |
| `punctuation` | `:leading-space`, `:trailing-newlines` | A run of characters that are not whitespace, letters or digits, with an optional leading space and trailing newlines |
| `newlines` | none | Whitespace up to its last newline |
| `whitespace` | none | A whitespace run; before text it leaves its last character for the next piece |

Letters, digits and whitespace follow Unicode categories.[^unicode] Rule options are checked when the tokenizer loads, so a misspelled option signals `tokenizer-error`. A mod adds a rule kind by specializing `rule-matcher`:

```lisp
(defmethod local:rule-matcher ((kind (eql :hash)) options)
  (lambda (text start) (and (char= (schar text start) #\#) (1+ start))))
```

### Inheritance

Rules merge by kind along the class precedence list. A child's rule replaces the parent's in place and keeps its position; a rule with a new kind goes at the end. `:model` and `:pre` merge the same way, so a child that lists `:pre` replaces the parent's names. Redefining a parent updates its children.

### Standard tokenizers

| Tokenizer | `tokenizer.ggml.pre` | Digits | `:ignore-merges` | `:add-bos` |
|---|---|---|---|---|
| `default-bpe` | missing, empty, `default` | staged Unicode runs and ASCII triples | no | no |
| `llama-bpe` | `llama-bpe`, `llama3`, `llama-v3`, `falcon3`, `falcon-h1`, `pixtral`, `midm-2.0`, `lfm2` | up to 3 | yes | yes |
| `qwen2` | `qwen2`, `deepseek-r1-qwen`, `kormo` | 1 | no | no |

All three use `tokenizer.ggml.model` `gpt2`. `qwen2` inherits `llama-bpe`, so it sets both options itself. `default-bpe` has its own [four-stage split](pre-tokenizer-stages.md#default-gguf-pipeline); it is not an alias for plain GPT-2 splitting.

## HF files

The optional Safetensors system reads the shipped Llama and Qwen3 `tokenizer.json` pipelines and special-token configuration. See [HF tokenizers](hf-tokenizers.md) for input validation and coverage.

## Errors

| Condition | Signalled when |
|---|---|
| `tokenizer-error` | Malformed definition or vocabulary, an invalid token id, `add-bos` with no BOS token, or several tokenizers matching one file |
| `unknown-tokenizer` | No tokenizer matches the file's model and `pre`; `unknown-tokenizer-model` and `unknown-tokenizer-pre` return them |

Loading signals `tokenizer-error` for a vocabulary that is not byte-level text, a merge that is not `"left right"` of known tokens, a missing byte token, a `token_type` list of the wrong length, and a special-token id outside the vocabulary.

## Limitations

| Missing | Tracker |
|---|---|
| Other `pre` types and SentencePiece models | [#33](https://github.com/communal-software/cl-inference/issues/33) |
| Placing a child's rule before an inherited one | [#34](https://github.com/communal-software/cl-inference/issues/34) |
| Chat templates | [chat templates](chat-templates.md#limitations) |

[^regex]: The flat standard rules reproduce llama.cpp's `llama3` and `qwen2` patterns, for example `\p{N}{1,3}` and `\s+(?!\S)`, as hand-written scanners. The tests compare both against the patterns run through Python's `regex` module.
[^merges]: The llama 3 merge list does not reach every token, so merging alone encodes some words differently from llama.cpp.
[^unicode]: Letter and number are the general categories `L*` and `N*`; whitespace is the `White_Space` property. A cache covers the Basic Multilingual Plane.
