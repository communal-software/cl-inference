# Gemma 3

`local:gemma3` runs **Gemma 3 270M IT** from Q8_0 GGUF on CPU/MLX and [official floating Safetensors](gemma-hf.md) on CPU/MLX through the standard `complete` contract. Text chat includes a leading system message, multi-turn history, streaming, sampling, stop strings, deadlines and cancellation.

```lisp
(asdf:load-system "cl-inference/quant")
(cl-inference/local:with-weights
    (weights "reference/models/gemma-3-270m-it-Q8_0.gguf")
  (let ((backend (cl-inference/local:make-local-backend weights)))
    (unwind-protect
         (cl-inference:complete backend
           :messages '((:role :system :content "Answer briefly.")
                       (:role :user :content "What is the capital of France?"))
           :temperature 0 :max-tokens 32)
      (cl-inference/local:close-local-backend backend))))
```

Keep weights open while the backend is in use. Sessions use F32 KV storage by default; `:cache-type :q8` is opt-in. Both session policies and exact-prefix reuse follow the [local backend](local-backend.md).

## Model behavior

| Area | Behavior |
|---|---|
| Architecture | 18 layers; embedding dimension 640; four query heads and one KV head; head dimension 256 |
| Attention | Five sliding layers then one global layer; local window 512 tokens |
| Position encoding | NeoX RoPE; theta 10,000 locally and 1,000,000 globally |
| Blocks | Scaled embeddings, Q/K RMSNorm, GELU-gated MLP, post-attention and post-MLP RMSNorm[^norms] |
| Tokenizer | GGUF `llama-spm`, prefix spaces disabled; HF [ranked character BPE](gemma-hf.md#tokenizer-pipeline); BOS 2, EOS 1 |
| Chat | `gemma3-chat`; leading system text prefixes the first user turn; user/assistant turns alternate |
| Reply stops | `<end_of_turn>`, `<eos>` and legacy `</s>` |
| Context | Model metadata declares 32,768; default session capacity is 4,096 |

The renderer matches the pinned checkpoint template for text. It trims each turn's text, renders assistant turns as `model`, and inserts BOS once according to tokenizer metadata. The legacy `</s>` alias is treated as a control/end token, matching the pinned reference despite its user-defined metadata type. Caller text keeps the shared [control-token boundary](chat-templates.md#message-data).

Tools, tool messages and image/audio content are rejected. Local layers use correct window masks over the session's full KV history; memory allocation and attention scoring still grow with context. See [Limitations](#limitations).

[HF preparation and acceptance](gemma-hf.md#acceptance) use independent Transformers F32 references. [MLX GGUF acceptance](mlx-gguf-acceptance.md) uses the Q8_0 artifact below.

## Pinned acceptance

The [manifest](../tests/acceptance/gemma-model.json) pins the public [ggml-org checkpoint](https://huggingface.co/ggml-org/gemma-3-270m-it-GGUF), its template, the F32-dequantized artifact and llama.cpp revision. Model files and oracle outputs stay in gitignored `reference/`.

```sh
curl -fL -o reference/models/gemma-3-270m-it-Q8_0.gguf \
  https://huggingface.co/ggml-org/gemma-3-270m-it-GGUF/resolve/e7647be17ae1108f2f605ed061ca0608b171afff/gemma-3-270m-it-Q8_0.gguf
python3 tests/acceptance/prepare-gemma.py
CL_INFERENCE_GEMMA_ACCEPTANCE_STRICT=1 \
sbcl --dynamic-space-size 8GB --non-interactive --eval '(require :asdf)' \
  --eval '(asdf:load-system "cl-inference/local/acceptance")' \
  --eval '(unless (fiveam:run! :cl-inference/gemma/acceptance) (uiop:quit 1))'
```

Preparation requires the pinned llama.cpp checkout described in [model acceptance](acceptance.md), CMake and a C++17 compiler. It verifies input hashes and clean inference sources, builds the CPU oracle and dequantizes the Q8_0 weights to F32. The download is 292 MB; the F32 reference is about 1.1 GB.

| Comparison | Required result |
|---|---|
| Prompt token IDs | Exact |
| All 18 layer outputs and final logits | Cosine ≥0.9999; relative error ≤0.02 |
| Top-five logit IDs | Exact order |
| Greedy output | Same IDs for up to eight tokens; matching finish reason |
| Chunked prefill and cached-decode logits | Same numerical bounds; 128-token prefill batches |

The five cases cover plain text, user chat, system-prefixed chat, multi-turn chat and a 557-token prompt crossing the sliding window. The strict suite passes 265 checks without skips on SBCL. Missing inputs skip ordinary runs; strict runs fail. Hash and provenance mismatches fail in either mode.[^reference]

## Tokenizer and template fixtures

```sh
uv run --with-requirements tools/chat-template-requirements.txt python tools/gemma-reference.py
uv run --with-requirements tools/chat-template-requirements.txt python tools/gemma-reference.py --check
```

The generator verifies model/template hashes and the reference revision. It compares 252 tokenizer cases against llama.cpp, including all four BOS/special-token combinations, and 20 text-template cases against pinned Jinja. Template cases run offline; real-vocabulary cases require the pinned model. Tiny offline models cover the architecture and client contract independently of downloads.

## Limitations

| Boundary | Tracker |
|---|---|
| Full-capacity KV storage and full-prefix scoring in local layers | [#70](https://github.com/communal-software/cl-inference/issues/70) |
| Gemma affine quantization and other GGUF variants | Outside the verified routes |
| Gemma 4 family | [#69](https://github.com/communal-software/cl-inference/issues/69) |
| Other Gemma 3 sizes, weight variants, tools and modalities | Outside the verified model and text contract |

[^norms]: GGUF conversion incorporates Gemma's `1 + weight` RMSNorm convention. GGUF runtime normalization uses stored weights directly. HF binding adds the offset once to model-owned norm vectors; the tied LM head shares embedding weights. The GGUF loader rejects unsupported logit softcapping and RoPE settings.
[^reference]: Acceptance uses F32-dequantized weights from the same Q8_0 file to isolate architecture correctness from llama.cpp's native activation quantization. It does not establish native-Q8 llama.cpp greedy equivalence or full-context performance.
