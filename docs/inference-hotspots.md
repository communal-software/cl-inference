# CPU inference hotspots

Encoded matrix multiplication dominates Qwen decode samples. The generic-copy profile also identifies prefill copying and allocation costs. [Bulk-copy measurements](https://github.com/communal-software/cl-tensor/blob/trunk/docs/copy-performance.md) report median paired prefill time reductions of 33.5% for F16 and 20.0% for Q8_0. The [SIMD comparison](simd-inference-performance.md) separately measures the strided primitive integration.

## Sampled costs

Qwen3 0.6B, 128 prompt tokens, serial CPU execution and one native thread:

| Format / phase | Samples | Main matrix entry: self samples | Other observed costs |
|---|---:|---|---|
| F16 prefill | 4,588 | BLAS GEMM entry: 38.7% | Zeroing 9.0%; large allocation 6.0%; generic offset traversal 27.7% inclusive |
| F16 cached decode | 4,273 | `encoded_tile1`: 54.9% | Zeroing 7.1%; float fill 6.8%; heap vector allocation 3.7% |
| Q8_0 prefill | 4,828 | `q8_0_matmul_tile8`: 59.1% | Zeroing 8.3%; large allocation 4.1%; generic offset traversal 18.3% inclusive |
| Q8_0 cached decode | 4,403 | `q8_0_matmul_tile4`: 58.7% | Float fill 8.7%; zeroing 7.2%; heap vector allocation 5.3% |

Inclusive traversal costs contain copying and child calls; they overlap other entries. Sample shares describe sampled stacks, rather than predicted speedups or exact elapsed-time fractions.

## Next measurements

1. Attribute weight conversion and loading costs within encoded matrix kernels. The [ARM64 wider-tile experiment](https://github.com/communal-software/trivial-simd/blob/trunk/docs/encoded-decode-tiles.md) does not establish a Qwen improvement; the four-output kernel remains selected. Compare complete decode calls before selecting another kernel change.
2. Attribute remaining copying and allocation after the bulk-copy path. The generic-copy profile records 18–28% inclusive traversal samples; reusable scratch requires separate ownership and layout validation.

The table describes the generic-copy runtime identified in the archived metadata. Bulk-copy results use a separate alternating-process comparison; the remaining candidates require further measurements.

## Method and evidence

Four fresh SBCL processes each load one pinned model, verify its hash, compute expected greedy tokens, prepare a 128-token prefix, warm up, and collect garbage before profiling. F16 and Q8_0 both produce expected IDs `(9625 374 785 6722)`. Cached decode overwrites the same next position. Prefill runs without KV caching.

SBCL CPU sampling requests a 2 ms interval over repeated inference calls for at least 30 seconds. The loops complete 9 F16 prefills, 201 F16 decodes, 6 Q8 prefills and 164 Q8 decodes. Model loading and expected-token preparation are outside the sampling window.[^reproduce]

[Reports, environment observations, exact harnesses and revision/hash metadata](benchmark-runs/2026-10-20-inference-hotspots/) preserve all four runs. System-wide swap-in counter increases across the complete processes are 24, 12, 17 and 8 pages respectively (16 KiB pages); swap-out counters do not increase. These observations include loading and setup, rather than isolating the sampling window.

## Limitations

This is a single Apple M1 with 8 GB RAM, macOS 15.7.5 and SBCL 2.6.8. Existing swap usage is about 5.5 GiB. Two unrelated CPU-bound jobs remain active. These are exploratory hotspot samples, not controlled throughput comparisons.

The macOS native `sample` utility stalls on these processes and is terminated manually. No separate native stack report is available. SBCL resolves foreign entry names but does not provide a full native call tree. Its local documentation warns that Darwin CPU sampling can deliver signals disproportionately across threads; collected samples represent about 9 seconds of nominal sampling time per 30-second wall-clock loop. Treat percentages as approximate leads.

The profile covers Qwen at one prompt length with automatic dispatch. It does not establish Llama, GPU, parallel executor, longer-context or generation-wide costs.

[^reproduce]: The archived Lisp harness reads `PROFILE_MODEL` (`QWEN-F16` or `QWEN-Q8`) and `PROFILE_PHASE` (`prefill` or `decode`). Run it with `VECLIB_MAXIMUM_THREADS=1 TRIVIAL_SIMD_BACKEND=native sbcl --dynamic-space-size 3072 --script <harness>`. It contains the measured checkout paths. The archived Python orchestrator also attempts the stalled native sampler and requires manual termination of that sampler; use the Lisp harness directly for SBCL reports. Revision and file hashes are in `metadata.json`.
