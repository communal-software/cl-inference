# HF SentencePiece BPE

The HF reader loads the pinned **TinyLlama-1.1B-Chat-v1.0** `tokenizer.json` pipeline through the existing tokenizer API. It requires no native tokenizer library or original SentencePiece `.model` file.[^artifact]

```lisp
(asdf:load-system "cl-inference/safetensors")
(cl-inference/local:with-weights (weights #p"models/TinyLlama-1.1B-Chat-v1.0/")
  (let ((tokenizer (cl-inference/local:load-tokenizer weights)))
    (list (cl-inference/local:encode tokenizer "Hello") ; => #(1 15043)
          (cl-inference/local:decode tokenizer #(1 15043))))) ; => "Hello"
```

The directory contains ordinary Safetensors weights and the pinned HF sidecars. Tokenizer parity does not establish real-model inference acceptance. The bundled chat template is [outside the supported template coverage](#limitations); callers supply an explicit native template.

[Gemma 3 270M IT](gemma-hf.md#tokenizer-pipeline) has a separate pinned character-BPE profile: no prepended space, no initial-space stripping, and different added tokens. Both profiles use the existing tokenizer API.

## Supported pipeline

| Stage | Behavior |
|---|---|
| Model | Character-based BPE, ranked merges, `<unk>`, `fuse_unk=true`, all 256 `<0xHH>` fallback tokens |
| Added tokens | `<unk>`, `<s>` and `</s>` are special; added-token matching follows the [shared contract](tokenizers.md#special-tokens) |
| Normalizer | Prepend `▁` to each nonempty ordinary span, then replace literal spaces with `▁` |
| Pre-tokenizer | None; `pre-tokenize` signals `tokenizer-error` |
| Postprocessor | One BOS for single-sequence input; BOS/EOS/unknown names and IDs must agree with the pipeline |
| Decoder | Replace `▁` with spaces, decode byte-fallback runs, fuse pieces, then strip one initial space |

Missing characters become UTF-8 byte tokens before ranked merging. Tabs, newlines and decomposed Unicode retain their spelling. This differs from [GGUF SentencePiece](sentencepiece.md), which merges by vocabulary score and preserves the synthetic leading space.

BOS defaults to enabled, including for empty input. `tokenizer_config.json` can disable it, and callers override it with `:add-bos`. Special-token spellings remain ordinary text unless `:special t` enables matching. Chat markup retains its provenance; caller text does not become control tokens.

`decode` skips controls unless `:special t` requests them. A rendered BOS prevents stripping a space that follows it: decoding `#(1 15043)` with `:special t` produces `"<s> Hello"`. `token-bytes` returns piece bytes before sequence-level stripping.

## Streaming

The streaming decoder buffers each consecutive byte-token run until an ordinary rendered token arrives or `detokenizer-finish` is called. Concatenated stream output equals batch decoding, including malformed bytes. Call `detokenizer-finish` to release a final byte run.

A valid run decodes as UTF-8. An invalid or incomplete run produces one U+FFFD per byte in the entire run. Space replacement precedes byte decoding, so byte tokens encoding `▁` preserve that character. Skipped special tokens do not interrupt a byte run.

## Validation

The reader validates the declared pipeline, vocabulary, merges, byte coverage, added-token options and special-token configuration. Unsupported stage settings, cleanup, padding, truncation and automatic EOS insertion signal `tokenizer-error`. Unknown chat templates signal `chat-template-error` when loaded separately.

Offline fixtures and the pinned real artifact compare exact IDs, decoding and streaming with Hugging Face Tokenizers. See [reference generation](testing.md#hf-sentencepiece-reference).

## Limitations

| Boundary | Tracker |
|---|---|
| Other HF SentencePiece pipelines, Unigram and original `.model` files | [#75](https://github.com/communal-software/cl-inference/issues/75) |
| TinyLlama's bundled chat template and other unpinned template variants | [Template limitations](chat-families.md#limitations) |
| Automatic EOS insertion | [#65](https://github.com/communal-software/cl-inference/issues/65) |

Paired inputs and arbitrary HF pipeline interpretation are outside this reader's coverage.

[^artifact]: Repository `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, revision `fe8a4ea1ffedaf415f4da2f062534de366a451e6`. `tools/hf-sentencepiece-manifest.json` pins the `tokenizer.json` and `tokenizer_config.json` SHA256 checksums and the development-only `tokenizers==0.21.4` oracle. The original `.model` file is not read.
