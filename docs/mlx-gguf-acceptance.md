# MLX GGUF acceptance

Strict acceptance runs all 15 plain/chat/thinking cases across Llama 3.2 1B Instruct and Qwen3 0.6B in Q4_0, Q4_K_M and Q6_K form. The reference executes each model's own packed weights after independent F32 dequantization in llama.cpp.

## Prepare

Use the six artifact hashes in `tests/acceptance/quant-models.json` and llama.cpp revision `8bb3cf0d216a5ae48360179f2931daf284be4d01`. Preparation requires the pinned F16 source models from [CPU acceptance](acceptance.md), Python 3.11+, CMake and a C++ compiler.

```sh
python3 tools/prepare-mlx-gguf.py
```

The generator reuses the [CPU quantization preparation](quant-acceptance.md), records prefill and cached-decode logits in little-endian F32, and writes `tools/mlx-gguf-reference-manifest.json`. Review generated manifest changes before trusting them. It pins model provenance, the oracle source hash and every consumed output file. Models and large dumps stay in gitignored `reference/`; allow the CPU preparation's approximately 9 GB of extra disk space.[^oracle]

## Run

```sh
CL_INFERENCE_MLX_GGUF_ACCEPTANCE_STRICT=1 \
  sbcl --dynamic-space-size 8GB --non-interactive --eval '(require :asdf)' \
  --eval '(asdf:test-system "cl-inference/mlx-gguf/acceptance")'
```

Missing required inputs fail strict mode; ordinary runs skip absent model/dump inputs. Hash and provenance mismatches fail in either mode. Skips do not establish acceptance.

## Verified run

On Apple M1/SBCL 2.6.8, all 15 cases pass 1,092 checks without skips. Across full/chunked prefill and cached-decode logit comparisons, the maximum relative L2 error is `0.0061835`, minimum cosine is `0.999982406`, and maximum absolute error is `0.0807533`. Prefill top-five IDs and all greedy IDs match.

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

Chunked prefill uses `n-batch 7`. The suite checks each of eight cached decode steps unless a stop token ends the reference earlier. Error reports include maximum absolute error, relative L2 and cosine. Cached non-greedy ranks may exchange positions when logits are closer than the numerical error bound. Native packed llama.cpp execution is not the numerical oracle because it introduces additional activation quantization.

[^oracle]: `DUMP_DECODE_LOGITS=1` enables the optional binary output in the development C++ oracle. Existing CPU trace and greedy files retain their formats. The last non-stop greedy token is decoded as well, yielding nine logit rows for an eight-token case. Oracle source changes require regeneration and manifest review.
