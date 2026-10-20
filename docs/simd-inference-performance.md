# SIMD inference performance

The strided SIMD integration shows no repeatable inference speedup in this CPU profile. In 22 of 24 cases, the median paired time change is below 0.4%. Llama F16 with 128 prompt tokens varies too widely to establish an improvement; see [Limitations](#limitations).

## Results

Positive percentages mean less time. Each cell is the median of five paired comparisons, not the ratio of the two timing medians.[^summary]

| Model | Prompt tokens | Prefill | Cached decode | Four-token generation |
|---|---:|---:|---:|---:|
| Llama 3.2 1B F16 | 16 | -0.16% | -0.05% | -0.15% |
| Llama 3.2 1B F16 | 128 | +9.68% | +0.31% | +15.73% |
| Llama 3.2 1B Q8_0 | 16 | +0.19% | +0.17% | -0.01% |
| Llama 3.2 1B Q8_0 | 128 | -0.13% | -0.02% | +0.01% |
| Qwen3 0.6B F16 | 16 | -0.02% | +0.05% | +0.13% |
| Qwen3 0.6B F16 | 128 | +0.13% | +0.16% | -0.22% |
| Qwen3 0.6B Q8_0 | 16 | -0.05% | -0.05% | +0.04% |
| Qwen3 0.6B Q8_0 | 128 | -0.11% | -0.01% | -0.16% |

No case is at least 5% faster in four of five pairs. Llama F16 at 128 tokens has prefill changes from −6.19% to +22.71%, and generation changes from −30.60% to +52.84%. Its generation medians are 18.46 seconds before and 19.54 seconds after, despite a positive median paired change. These results do not support a whole-model speedup claim.

## Method and reproduction

Five fresh-process pairs alternate before/after order. Each process measures all four models at 16 and 128 prompt tokens, using automatic dispatch, no tensor executor and `VECLIB_MAXIMUM_THREADS=1`. Both SIMD revisions use fresh Release builds from Git archives and separate ASDF caches.

| Dependency | Revision |
|---|---|
| trivial-simd baseline | `cd61d43c2986c05ac32503d7619b2e7b67d1b584` |
| trivial-simd integrated | `26bb84d9afaeb146b5e92c1b27f594268914e159` |
| cl-inference runtime | `575dad50c60fa64ef7c0995c5bce7385dae6095b` |
| cl-tensor | `ad606e270c74f5ad8f7a65959c98555467d65841` |

Run from the repository with SBCL, CMake, the sibling trivial-simd checkout containing both revisions, and the pinned model fixtures used by `tools/parallel-bench.lisp`:

```sh
python3 tools/simd-inference-profile.py /tmp/simd-inference.csv
```

The runner rejects sustained unrelated CPU load. Each case reports the median of three calls after one warmup, with full collection before each call. Prefill excludes KV caching; decode overwrites one position after a prepared prefix; generation resets the session and includes prompt processing plus four output tokens. Loading, model hash checks and expected-token preparation are outside timing.

All 240 timing rows pass validation. Every measured generation returns its expected four tokens, and expected IDs match across all ten processes. The local inference suite passes 39,739 checks.

## Evidence

- [Raw timings](benchmark-runs/2026-10-20-simd-inference.csv)
- [Build and verification metadata](benchmark-runs/2026-10-20-simd-inference.json)
- [Paired summary](benchmark-runs/2026-10-20-simd-inference-summary.json)
- [Memory observation](benchmark-runs/2026-10-20-simd-inference-memory.json)
- [Process logs](benchmark-runs/2026-10-20-simd-inference-logs/)

## Limitations

This is one Apple M1 with 8 GB RAM, macOS 15.7.5 and SBCL 2.6.8. An observation during the first baseline process records 6.66 GiB of swap use. Alternating order and the CPU guard do not remove memory pressure. All samples, including outliers, remain in the evidence.

The comparison covers the complete integration, including shared pointwise dispatch changes. It does not isolate individual admitted SIMD kernels or establish GPU, executor, other-machine, native-default threading or long-context performance. Primitive-level speedups do not establish whole-model gains. The [Qwen hotspot profile](inference-hotspots.md) identifies matrix kernels and allocation/copying as investigation candidates; controlled comparisons still require sufficient free memory.

[^summary]: The paired change is `100 × (1 − after/before)`. Per-version times are medians across five processes; their ratio can differ from the median paired change. Raw allocation counts cover Lisp allocation, excluding native allocation. Metadata records the exact benchmark source hashes alongside runtime revisions.
