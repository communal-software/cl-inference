# Gemma HF execution

**Gemma 3 270M IT** runs from official floating Safetensors on CPU and MLX. Both routes compute in F32 and use the same pinned HF tokenizer, text-chat renderer and session API.[^artifact]

## Run

Accept the model's Hugging Face terms and authenticate with `hf auth login` or set `HF_TOKEN`. Preparation downloads only pinned, hash-verified files:

```sh
uv run --with-requirements tools/gemma-hf-requirements.txt \
  python tools/gemma-hf-reference.py --download --inference
```

```lisp
(asdf:load-system "cl-inference/safetensors")
(cl-inference/local:with-weights (weights #p"reference/gemma-hf/model/")
  (let ((backend (cl-inference/local:make-local-backend weights)))
    (unwind-protect
         (cl-inference:complete backend
           :messages '((:role :user :content "What is the capital of France?"))
           :temperature 0 :max-tokens 8)
      (cl-inference/local:close-local-backend backend))))
```

For MLX, build the native libraries described in [MLX setup](mlx.md#setup), load `cl-inference/mlx`, and use `make-mlx-backend` / `close-mlx-backend` in the same lifetime. Both backends borrow open weights. The separate `chat_template.jinja` selects the existing Gemma text renderer.

## Tokenizer pipeline

The pinned `tokenizer.json` completely describes this **ranked character-BPE** pipeline. Neither Unigram nor a native SentencePiece library is involved.

| Stage | Behavior |
|---|---|
| Normalizer | Replace literal spaces with `▁`; no prepended space |
| Pre-tokenizer | Split literal spaces, merged with the preceding span, after normalization |
| Model | Ranked BPE; `<unk>`; all 256 byte-fallback tokens |
| Added tokens | Seven special controls; 6,408 ordinary added tokens, including newline runs |
| Postprocessor | BOS 2 by default; EOS 1 and unknown 3 |
| Decoder | Replace `▁`, decode byte-fallback runs, fuse; no initial-space stripping |

`encode` accepts existing `:add-bos` and `:special` options. Caller text matches controls only with `:special t`; ordinary added tokens always match. Streaming buffers consecutive byte tokens and flushes them at the next rendered non-byte token or `detokenizer-finish`. Concatenated streaming output equals batch decoding, including malformed bytes.

## Model behavior

HF `gemma3_text` configuration supplies the local/global layer pattern, 512-token window, local/global RoPE bases and explicit 256-element head dimension. Embeddings scale by F32 `sqrt(640)`. Attention uses Q/K norms, and the MLP uses tanh-approximate GELU. Post-attention and post-MLP norms apply before residual addition.

HF norm vectors become model-owned F32 `1 + weight` vectors once during binding. Source tensors retain their stored values. GGUF norm vectors already contain the offset and remain unchanged. The official checkpoint uses tied embeddings/output weights even though its configuration omits `tie_word_embeddings`.

The tokenizer retains `<image_soft_token>` at ID 262144 beyond the 262144 embedding rows. MLX accepts this single Gemma control-token declaration; attempting to execute that ID still signals a session error. Image content remains outside the text contract.

## Acceptance

```sh
CL_INFERENCE_GEMMA_HF_STRICT=1 sbcl --dynamic-space-size 8GB --non-interactive \
  --eval '(require :asdf)' \
  --eval '(asdf:load-system "cl-inference/local/acceptance")' \
  --eval '(unless (fiveam:run! :cl-inference/gemma-hf/acceptance) (uiop:quit 1))'
```

For MLX, load `cl-inference/mlx/acceptance` and run `:cl-inference/gemma-hf-mlx/acceptance` through `fiveam:run!` with the same strict environment variable.

| Comparison | Requirement |
|---|---|
| Token IDs and decoded text | Exact pinned Tokenizers parity; batch/stream equality |
| All 18 layer outputs at the final prompt position | Finite; cosine ≥0.9999; relative L2 ≤0.02 |
| Full/chunked prefill and cached-decode logits | Same numerical bounds |
| Prefill top-five IDs | Exact order |
| Greedy generation | Exact IDs up to eight tokens and matching finish reason |
| Public chat | Matching text, streamed text and finish reason |

Strict acceptance passes 456 CPU checks and 457 MLX checks without skips on M1/SBCL. The independent PyTorch/Transformers oracle runs official BF16 storage converted to F32, using eager attention. Cases cover plain text, user chat, system-prefixed chat, multi-turn chat and a 557-token prompt crossing the local window. Missing inputs skip ordinary runs and fail strict runs. Hash or provenance mismatches always fail. Tiny fixtures cover F32/F16/BF16 storage, tied heads and norm offsets independently of downloads.

The manifest pins inputs, outputs, reference packages and generator source. `--check` verifies fixtures and existing output hashes; `--check --inference` also regenerates and compares the inference outputs.

## Limitations

| Boundary | Tracker |
|---|---|
| Other HF SentencePiece pipelines, Unigram and original `.model` loading | [#75](https://github.com/communal-software/cl-inference/issues/75) |
| Gemma affine quantization and other model sizes/modalities | Outside the verified routes |
| Full-prefix local attention and full-capacity caches | [#70](https://github.com/communal-software/cl-inference/issues/70) |

[^artifact]: `google/gemma-3-270m-it`, revision `ac82b4e820549b854eebf28ce6dedaf9fdfa17b3`. `tools/gemma-hf-manifest.json` pins SHA256 hashes. Reference-only dependencies are PyTorch 2.10.0, Transformers 5.5.0 and Tokenizers 0.22.2; inference has no Python runtime dependency. Weight storage is approximately 536 MB; the independent F32 model requires additional memory.
