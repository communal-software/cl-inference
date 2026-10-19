# MLX GGUF acceptance

Strict MLX GGUF acceptance covers 20 pinned cases: 15 Llama 3.2 1B Instruct/Qwen3 0.6B Q4/Q6 cases and five Gemma 3 270M IT Q8_0 cases. References use each model’s own packed weights independently dequantized to F32 by llama.cpp.

## Gemma Q8_0

Prepare the pinned GGUF using [Gemma acceptance](gemma.md#pinned-acceptance). `tools/prepare-mlx-gguf.py` generates references and manifests for every packed route, including Gemma. The committed manifest pins the input model, oracle source and consumed output hashes.

```sh
CL_INFERENCE_MLX_GGUF_ACCEPTANCE_STRICT=1 \
sbcl --dynamic-space-size 8GB --non-interactive --eval '(require :asdf)' \
  --eval '(asdf:load-system "cl-inference/mlx-gguf/acceptance")' \
  --eval '(unless (cl-inference/mlx-gguf/tests:run-acceptance :strict t :suite :cl-inference/gemma-mlx-gguf/acceptance) (uiop:quit 1))'
```

The five cases cover plain text, user chat, a system prefix, multi-turn chat and a 557-token window-crossing prompt. All 311 checks pass without skips on M1/SBCL. Full/chunked prefill, every layer's last prompt position, cached decoding and greedy generation meet the requirements below. Gemma uses 128-token prefill batches; other cases use seven-token batches.

## Prepare

The full preparation also requires the Gemma Q8_0 download described above. Use the six Llama/Qwen artifact hashes in `tests/acceptance/quant-models.json` and llama.cpp revision `8bb3cf0d216a5ae48360179f2931daf284be4d01`. Preparation requires the pinned F16 source models from [CPU acceptance](acceptance.md), Python 3.11+, CMake and a C++ compiler.

```sh
python3 tools/prepare-mlx-gguf.py

# Regenerate only the 15 Llama/Qwen cases
python3 tools/prepare-mlx-gguf.py --family llama --family qwen
```

Repeat `--family llama|qwen|gemma` to select families. Omission selects all three; unselected manifest entries and reference dumps are preserved.

The generator reuses the [CPU quantization preparation](quant-acceptance.md), records prefill and cached-decode logits in little-endian F32, and writes `tools/mlx-gguf-reference-manifest.json`. Review generated manifest changes before trusting them. It pins model provenance, the oracle source hash and every consumed output file. Models and large dumps stay in gitignored `reference/`; allow the CPU preparation's approximately 9 GB of extra disk space.[^oracle]

## Run

```sh
CL_INFERENCE_MLX_GGUF_ACCEPTANCE_STRICT=1 \
  sbcl --dynamic-space-size 8GB --non-interactive --eval '(require :asdf)' \
  --eval '(asdf:test-system "cl-inference/mlx-gguf/acceptance")'
```

All 1,403 checks pass on M1/SBCL without skips.

Missing required inputs fail strict mode; ordinary runs skip absent model/dump inputs. Hash and provenance mismatches fail in either mode. Skips do not establish acceptance.

## Requirements

| Check | Bound |
|---|---|
| Rendered prompt token IDs | Exact |
| Every layer after full prefill | Finite; cosine ≥0.9999; relative L2 ≤0.02 |
| Full/chunked prefill logits | Same numerical bounds; ordered top-five IDs match |
| Cached-decode logits | Same numerical bounds |
| Greedy IDs | Exact, up to eight tokens |
| Public generation | IDs before the first stop and finish reason match |
| Independent block/projection/embedding fixtures | Elementwise error ≤ `1e-4 + 1e-4 × abs(reference)` |

Chunked prefill uses the batch sizes described above. The suite checks each of eight cached decode steps unless a stop token ends the reference earlier. Error reports include maximum absolute error, relative L2 and cosine. Cached non-greedy ranks may exchange positions when logits are closer than the numerical error bound. Native packed llama.cpp execution is not the numerical oracle because it introduces additional activation quantization.

## Limitations

Acceptance covers only the pinned model artifacts and prompt cases above. Other variants and quantization formats remain unverified.

[^oracle]: `DUMP_DECODE_LOGITS=1` enables the optional binary output in the development C++ oracle. Existing CPU trace and greedy files retain their formats. The last non-stop greedy token is decoded as well, yielding nine logit rows for an eight-token case. Preparation verifies the pinned llama.cpp revision and rejects modified inference sources. Oracle source changes require regeneration and manifest review.
