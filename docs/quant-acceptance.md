# Q4/Q6 model acceptance

The suite checks Llama 3.2 1B Instruct and Qwen3 0.6B in Q4_0, Q4_K_M and Q6_K form. Each runs plain completion and chat; Qwen also runs thinking chat. All 15 cases must run to establish acceptance.

## Prepare inputs

Place the SHA256-verified F16 models from [Model acceptance](acceptance.md) in `reference/models/`. With Python 3.11+, Git, CMake and a C++ compiler:

```sh
python3 tests/acceptance/prepare-quant.py --native
```

The script builds llama.cpp at commit `8bb3cf0d216a5ae48360179f2931daf284be4d01`, quantizes those F16 models, verifies the generated models against [`quant-models.json`](../tests/acceptance/quant-models.json), and dumps both F32-dequantized and native packed execution. `--family llama|qwen` and `--format q4_0|q4_k|q6_k` restrict setup to one model or format. Native dumps are optional when `--native` is omitted.

Models, build products, logs and oracle dumps stay under gitignored `reference/`. Setup reuses `reference/quant-build/f32-scratch.gguf` for one dequantized model at a time; allow about 9 GB of extra disk space for all six packed models and that scratch file.

## Run strictly

```sh
CL_INFERENCE_QUANT_ACCEPTANCE_STRICT=1 \
sbcl --dynamic-space-size 16GB --non-interactive --eval '(require :asdf)' \
  --eval '(asdf:test-system "cl-inference/local/acceptance")'
```

Ordinary runs skip absent quantized inputs. Strict mode fails on missing models or dumps; model hash mismatches fail in both modes. Oracle provenance must match the model hash, source F16 hash, llama.cpp commit and dtype; native packed dumps cannot substitute for F32-dequantized dumps.

## Required comparisons

| Check | Bound |
|---|---|
| Prompt IDs | Exact |
| Every layer and final logits | Cosine ≥0.9999; relative error ≤0.02 |
| Top-five logits | Same IDs in order |
| Greedy generation | Same IDs for up to eight tokens and matching finish reason |
| Model tensors | Requested dtype present; Q4_K_M also contains Q6_K; all stored types supported |

The reference dequantizes each model's own packed weights to F32 before inference. It measures implementation agreement without llama.cpp's additional activation quantization. Native packed dumps remain available for separate drift measurements.[^blocks][^drift]

[^blocks]: Offline fixtures also compare all decoded elements with llama.cpp. Regenerate them with `python3 tests/local/generate-quant-fixtures.py` after building the pinned reference; routine tests do not need llama.cpp installed.
[^drift]: `python3 tests/acceptance/quant-drift.py` compares the native and F32-dequantized dumps. Across these prompts, native execution's maximum layer relative error is 0.030–0.051 for Llama and 0.064–0.110 for Qwen. Native Qwen Q4_K_M and Q6_K greedy IDs differ in some cases; this comparison does not gate Lisp acceptance.
