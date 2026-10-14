# Hugging Face tokenizers

`cl-inference/safetensors` loads the shipped Llama and Qwen3 tokenizer pipelines from sibling HF files. This support serves both CPU and MLX backends.

```lisp
(cl-inference/local:with-weights (weights #p"models/Qwen3-0.6B/")
  (let ((tokenizer (cl-inference/local:load-tokenizer weights)))
    (cl-inference/local:encode tokenizer "Hello!")))
```

## Files and pipelines

| Input | Behavior |
|---|---|
| `tokenizer.json` | Required; BPE vocabulary, merges, added tokens and supported split/byte-level pipeline |
| `tokenizer_config.json` | Special tokens, BOS policy and chat-template source |
| HF model configuration | EOS fallback and architecture template fallback |
| Llama pipeline | Llama byte-level splitting, ignore-merges behavior and optional BOS postprocessor |
| Qwen3 pipeline | NFC normalization and Qwen byte-level splitting |

Merges accept strings or two-element arrays. Added-token IDs must agree with vocabulary entries; gaps and conflicting IDs are rejected. Added tokens with word-boundary, trimming or normalization options are unsupported.

A Llama postprocessor adds only one BOS token to single-sequence inputs. Its name and ID must match the vocabulary and configuration. Unsupported normalizers, pre-tokenizers, decoders, padding, truncation and postprocessors signal `tokenizer-error`.

## Chat templates

Known template markers select the existing Llama or Qwen3 renderer. An explicitly supplied unknown template signals `chat-template-error`; an absent template uses the architecture's declared fallback. The reader does not execute Jinja.

Token IDs, Unicode normalization and rendered prompts are checked against the pinned Qwen acceptance reference. [Tokenizer operations](tokenizers.md) and [chat rendering](chat-templates.md) describe the shared APIs.

## Limitations

| Deferred capability | Tracker |
|---|---|
| Other HF tokenizer pipelines and SentencePiece | [#33](https://github.com/communal-software/cl-inference/issues/33) |
| Additional templates and stricter detection | [#38](https://github.com/communal-software/cl-inference/issues/38) |
| Control-token spelling in message text | [#37](https://github.com/communal-software/cl-inference/issues/37) |

Paired inputs and arbitrary Jinja execution are outside this reader's coverage.
