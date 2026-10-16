# Pre-tokenizer stages

`:pre-tokenizer-stages` splits text in sequential passes. Each pass subdivides the pieces from the previous pass; earlier boundaries remain intact. Use it when a tokenizer needs several splitting patterns rather than one ordered set of alternatives.

```lisp
(local:define-tokenizer numbered-words (local:byte-level-bpe)
  (:pre-tokenizer-stages
   (words (letters :prefix :space))
   (numbers (digits :max 2)))
  (:format :gguf (:model "gpt2") (:pre "numbered-words")))
```

With a matching vocabulary, `pre-tokenize` of `"hello 1234"` returns `("hello" " " "12" "34")`. Both `pre-tokenize` and `encode` use the same pipeline.

## Matching

| Behavior | Contract |
|---|---|
| Stage order | Declaration order |
| Rule order | First matching rule at each position |
| Unmatched text | Each contiguous gap remains a piece |
| Boundaries | A rule sees only its current piece, including for lookahead |
| Empty input | No pieces |
| Invalid matches | Non-integer, non-advancing or out-of-bounds matches signal `tokenizer-error` |

A stage contains a symbol name and at least one ordinary [rule](tokenizers.md#rules). Existing `rule-matcher` extensions work in stages without changing their `(text start)` calling convention. Staged definitions preserve unmatched text; flat definitions retain their error when no rule matches.

## Inheritance

Stages merge by name, including names from different packages. A child replaces an inherited stage's complete rule list without moving it. A new stage appends. Redefining a parent updates the effective definitions of its children; loaded tokenizers keep their compiled matchers.

Rules inside a stage accept [`:before` and `:after`](tokenizers.md#inheritance). Each stage resolves its own rule list from empty in declaration order; anchors must appear earlier in that stage. Placement does not reorder stages or merge a replacement stage with inherited rules.

`tokenizer-pre-stages` returns the effective ordered alist of stage names and rules for a tokenizer class, class name or instance. `tokenizer-pre-rules` returns flat rules; it returns NIL for a staged tokenizer.

A class lineage cannot combine `:pre-tokenizer` and `:pre-tokenizer-stages`. Loading such a tokenizer signals `tokenizer-error`. Malformed stage declarations fail when `define-tokenizer` expands; rule options are checked when loading.

## Default GGUF pipeline

`default-bpe` declares four stages matching llama.cpp's default BPE preprocessing:[^reference]

| Stage | Split |
|---|---|
| `symbols` | Runs of Unicode punctuation and `$+<=>^~\|` |
| `words` | Case-sensitive contractions, optional-space letters/numbers/punctuation, and whitespace |
| `numbers` | Unicode number runs |
| `triples` | Exactly three ASCII digits per match; unmatched text stays together |

For example, `" 1234567"` splits as `(" " "123" "456" "7")`. `"123٤56789²345"` splits as `("123" "٤" "567" "89²" "345")`. The pipeline uses normal BPE merges and defaults to no BOS; metadata and caller BOS overrides retain their usual behavior.

[^reference]: Pinned llama.cpp revision `8bb3cf0d216a5ae48360179f2931daf284be4d01`. The offline fixtures use the actual C++ splitter and tokenizer; see [reference generation](testing.md#default-tokenizer-reference). This fallback does not infer a missing model-specific pre-type.
