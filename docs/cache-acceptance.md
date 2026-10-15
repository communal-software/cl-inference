# Cache acceptance

Cache checks compare prefix reuse with fresh execution, Q8 logits with F32, and Q8 execution with an independent Python MLX-LM implementation. All required inputs must be present; skips do not establish acceptance.

## Pinned inputs

| Execution | Models and weights |
|---|---|
| CPU | The pinned Llama 3.2 1B Instruct and Qwen3 0.6B F16 GGUF files in [CPU acceptance](acceptance.md) |
| MLX | The pinned Llama and Qwen original BF16 and affine 4-bit directories in [MLX acceptance](mlx-acceptance.md) |

`tools/kv-cache-reference-manifest.json` records hashes, package versions, the codec identifier and input provenance. Generated logits stay in gitignored `reference/*-kv-q8.json`. CPU oracles load the exact GGUF weights into the Python model and preserve GGUF RoPE ordering; comparison does not substitute HF weights.

## Generate references

Use the [pinned Python MLX environment](mlx-acceptance.md#reference-inputs). CPU reference generation also requires the existing llama.cpp checkout at `b11574` for its GGUF reader. Generation uses HF configuration and prompt IDs from the pinned MLX artifacts, even for CPU cases; running CPU acceptance only requires its pinned GGUF and generated oracle.

```sh
reference/mlx/venv/bin/python tools/kv-cache-reference.py
# Or regenerate one reviewed case:
reference/mlx/venv/bin/python tools/kv-cache-reference.py --case cpu-qwen
```

Generation updates the manifest hashes. Review changes to hashes and provenance before accepting regenerated fixtures.

## Run

```sh
CL_INFERENCE_CACHE_ACCEPTANCE_STRICT=1 \
  sbcl --dynamic-space-size 16GB --non-interactive --eval '(require :asdf)' \
  --eval '(asdf:test-system "cl-inference/local/acceptance")'

CL_INFERENCE_CACHE_ACCEPTANCE_STRICT=1 CL_INFERENCE_MLX_ACCEPTANCE_STRICT=1 \
  sbcl --dynamic-space-size 8GB --non-interactive --eval '(require :asdf)' \
  --eval '(asdf:test-system "cl-inference/mlx/acceptance")'
```

Run memory-heavy model checks separately.

## Checks

| Check | Requirement |
|---|---|
| Q8 versus independent Q8 oracle, prefill and eight decode steps | Finite logits; maximum absolute error `<1.0`, relative L2 `<0.05`; exact oracle greedy IDs |
| Q8 versus F32 at the same token positions | Relative L2 `≤0.15`, cosine `≥0.99` |
| F32 prompt replay and appended-token equivalence | Maximum absolute error `<0.005`, relative L2 `<0.0001` |
| Public Q8 prompt replay and generation | Eight oracle IDs; cached count excludes the replayed prompt token |
| Retained Q8 payload on the pinned models | Less than 30% of F32 |

Quality bounds apply to the selected prompts and model variants. Q8 rounding can amplify small arithmetic differences between engines and batch layouts. Weight quantization and KV quantization have separate reference checks.

Offline fixtures cover changed and shortened prefixes, generated-token reuse, disabled matching, malformed options, context overflow, reset, interruption, partial groups, ties-to-even, non-finite input, CPU/MLX codec agreement, threaded CPU attention and MLX array ownership.

The recorded acceptance runs pass 1,603 CPU checks and 376 strict MLX checks without skips. Across prefill and eight decode steps:

| Execution | Worst relative L2 versus Q8 oracle | Worst relative L2 versus F32 | Lowest cosine versus F32 |
|---|---:|---:|---:|
| CPU, two F16 models | 0.038158 | 0.135099 | 0.99265698 |
| MLX, four weight variants | 0.031786 | 0.134873 | 0.99275403 |

Native Q8 generation matches the eight independent-oracle greedy IDs in all six cases. Those six reference sequences also match their corresponding F32 references; other prompts can produce different output.

## Measure

The benchmark warms the model, then records three runs of fresh prefill, identical-prompt replay, a 64-token extension, and replay plus eight decode steps. It appends CSV rows: engine, model, cache type, trial, operation, prompt tokens, skipped prefix tokens, seconds and retained cache bytes.

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  /usr/bin/time -l sbcl --dynamic-space-size 8GB --script tools/benchmark-cache.lisp \
  cpu reference/models/Qwen3-0.6B-f16.gguf q8 /tmp/cache.csv 256 1024

/usr/bin/time -l sbcl --dynamic-space-size 8GB --script tools/benchmark-cache.lisp \
  mlx reference/mlx/qwen-float/ q8 /tmp/cache.csv 256 1024
```

Repeat with `f32` to compare storage and complete-call timing. Run each process separately. Process peak memory includes weights, runtime, cache and scratch; retained cache bytes count only K/V payload and scales. Neither quantity alone describes total device memory.

### Recorded profile

The [48 timing rows](benchmark-runs/2026-10-15-kv-cache.csv) and [process memory observations](benchmark-runs/2026-10-15-kv-cache-memory.csv) use M1/SBCL 2.6.8, context capacity 1024 and prefill batch size 512. CPU BLAS uses one native thread and no Lisp executor. Each cell is the median of three runs. The prompts repeat token ID 42 on the pinned Qwen3 0.6B models; these are execution measurements, separate from the chat-quality probes.

| Operation | CPU F32 | CPU Q8 | MLX F32 | MLX Q8 |
|---|---:|---:|---:|---:|
| Fresh 256-token prefill | 7.615 s | 8.016 s | 0.396 s | 0.413 s |
| Repeat prompt, replaying one token | 0.149 s | 0.697 s | 0.145 s | 0.226 s |
| Extend by 64 tokens | 2.180 s | 2.820 s | 0.171 s | 0.261 s |
| Replay plus eight generated tokens | 1.383 s | 7.643 s | 1.438 s | 2.027 s |

Retained K/V payload is **224 MiB for F32 and 63 MiB for Q8** on both engines. Q8 retains 71.875% fewer payload bytes, while CPU generation in this profile takes about 5.5 times as long.

| Process observation | CPU F32 | CPU Q8 | MLX F32 | MLX Q8 |
|---|---:|---:|---:|---:|
| Maximum RSS | 2043 MiB | 1890 MiB | 2955 MiB | 2654 MiB |
| macOS peak memory footprint | 901 MiB | 747 MiB | 5047 MiB | 2826 MiB |

RSS and macOS footprint account for mappings and device allocations differently. Peak process memory depends on the workload, allocator and temporary buffers; retained cache savings alone do not determine it.

## Limitations

| Ceiling | Tracker |
|---|---|
| Scalar Q8 conversion and full-prefix F32 attention scratch | [#57](https://github.com/communal-software/cl-inference/issues/57) |
| MLX full-capacity cache writes | [#49](https://github.com/communal-software/cl-inference/issues/49) |
