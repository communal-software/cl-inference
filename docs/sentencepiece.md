# Llama SentencePiece

`llama-spm` loads GGUF Llama SentencePiece vocabularies without an external tokenizer library. Encoding matches the pinned llama.cpp tokenizer; decoding preserves token-piece spaces.[^reference]

```lisp
(asdf:load-system "cl-inference/gguf")
(cl-inference/local:with-weights
    (weights "reference/llama.cpp/models/ggml-vocab-llama-spm.gguf")
  (let* ((tokenizer (cl-inference/local:load-tokenizer weights))
         (ids (cl-inference/local:encode tokenizer "Hello" :add-bos nil)))
    (cl-inference/local:decode tokenizer ids))) ; => " Hello"
```

The vocabulary-only reference file is enough for this example. The loaded tokenizer remains usable after its weights close.

## Selection and metadata

| Input | Behavior |
|---|---|
| `tokenizer.ggml.model = "llama"` | Selects `llama-spm`; `pre` does not affect selection |
| `tokens`, `scores`, `token_type` | Required, equally sized arrays; scores are finite F32 values |
| Byte tokens | All 256 `<0xHH>` entries are required with byte token type |
| `add_space_prefix` | Defaults to true |
| `add_bos_token` | Defaults to true; callers override it with `:add-bos` |
| BOS / EOS / unknown IDs | Metadata overrides defaults 1 / 2 / 0; IDs must fit the vocabulary |
| BPE merges | Not required or used |

Malformed vocabularies signal `tokenizer-error`. `tokenizer-ignore-merges-p` returns NIL. `pre-tokenize` signals `tokenizer-error`: this tokenizer merges characters directly rather than splitting text with BPE rules.

## Text and streaming

Adjacent symbols merge by descending vocabulary score, with leftmost ties. Unmerged characters absent from the vocabulary become UTF-8 byte tokens.

Encoding adds a prefix space to nonempty ordinary text at the beginning and after recognized special tokens when `add_space_prefix` is true. Literal spaces become `▁`; tabs, newlines and Unicode combining characters retain their spelling. Empty input follows the shared [BOS policy](tokenizers.md#functions).

Normal token decoding replaces `▁` with spaces. Byte tokens contribute raw bytes. `decode` and the streaming detokenizer preserve every token-piece space, including the synthetic leading space. They do not promise `decode(encode(text)) = text` when prefix spaces are enabled. Streaming buffers incomplete UTF-8 characters as described in [Tokenizers](tokenizers.md#streaming).

The shared [special-token contract](tokenizers.md#special-tokens) applies: user-defined tokens always match; control and unknown token spellings match only with `:special t`.

## Limitations

| Boundary | Tracker |
|---|---|
| Other GGUF SentencePiece variants | [#62](https://github.com/communal-software/cl-inference/issues/62) |
| HF SentencePiece pipelines beyond [TinyLlama BPE](hf-sentencepiece.md) | [#74](https://github.com/communal-software/cl-inference/issues/74) |
| Automatic EOS insertion | [#65](https://github.com/communal-software/cl-inference/issues/65); enabled configurations signal `tokenizer-error` |
| Extra-whitespace normalization | [#66](https://github.com/communal-software/cl-inference/issues/66); enabled configurations signal `tokenizer-error` |

Pinned tokenizer parity does not establish real-model inference or additional chat-template support. [Support](support.md#local-architectures) lists verified models.

[^reference]: llama.cpp revision `8bb3cf0d216a5ae48360179f2931daf284be4d01`, using its Llama SentencePiece vocabulary and synthetic score fixtures. The reference compares IDs and `llama_token_to_piece` bytes with zero leading-space stripping. See [reference generation](testing.md#sentencepiece-reference).
