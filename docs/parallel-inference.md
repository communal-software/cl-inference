# Parallel CPU inference

Bind `ct:*executor*` around a model or client call to schedule tensor operations and independent attention heads. NIL runs serially. The caller owns the executor and worker pool.

```lisp
(let ((ct:*executor* executor))
  (local:model-forward model '(3 0 5)))
```

The same binding applies to `session-prefill`, `generate`, and the CPU local backend. It does not schedule MLX device operations.

## Executor contract

| Responsibility | Contract |
|---|---|
| Interface | `(executor count callback)` calls `(callback start end)` |
| Partitioning | Disjoint integer ranges covering `[0,count)` exactly once; empty ranges allowed |
| Completion | Wait for every callback before returning or propagating a worker error |
| Policy | Caller chooses workers, partition size and serial thresholds |
| Lifetime | Weights, tensors, sessions and the pool remain valid until the call finishes |

Bounds are checked. Coverage and disjointness belong to the executor. A single head runs directly; zero heads schedule no work. Head callbacks inherit backend/FMA settings and disable nested tensor scheduling. Results retain head order, and cache writes finish before head work starts.[^tensor]

The [pool adapter example](../examples/executor.lisp) accepts a synchronous pool runner whose jobs receive worker ID and worker count:

```lisp
(let ((ct:*executor* (cl-inference/examples:pool-executor pool #'run-pool)))
  (local:generate session tokens :max-tokens 8))
```

`run-pool` must join all jobs before reporting errors. Cancellation and deadlines retain the existing checkpoints between inference passes; callbacks are not interrupted. No callback remains active when an operation returns or signals.

## Measurement

Run measurements separately from tests. The benchmark reports median complete-call time, Lisp allocation and generation throughput for serial, two-worker and four-worker execution.[^measurement] Synthetic cases cover projections in floating/packed formats and GQA attention. Real cases use pinned Llama 3.2 1B and Qwen3 0.6B F16/Q8_0 models with 16- and 128-token prompts.

```sh
VECLIB_MAXIMUM_THREADS=1 \
  sbcl --dynamic-space-size 3GB --script tools/parallel-bench.lisp
```

Set `CL_INFERENCE_PARALLEL_SYNTHETIC_ONLY=1` to omit real models. Compare native threading settings in separate processes. [Half-precision dispatch](half-linear-performance.md) uses encoded execution for a single activation row with directly readable weight rows on the measured profile; other executor-bound batches use tiles.

## Measured profile

The scheduling reference profile measures attention and row scheduling on M1/SBCL. Four workers reduce synthetic attention time by about 1.6–2.1× and improve longer F16 prefills. Its `baseline-auto` configuration uses serial automatic dispatch and executor-bound tiled half-weight calls. The [half-weight dispatch profile](#half-weight-dispatch-profile) measures current automatic decode separately. Keep serial execution as the default; choose an executor from measurements of the intended format and phase.[^profile]

The reference table uses a single native BLAS thread and `baseline-auto` dispatch. Each cell shows **serial → four workers**. Generation includes prefill and four completed tokens.

| Model | Prompt tokens | Prefill, ms | Decode, ms | Generation, tokens/s |
|---|---:|---:|---:|---:|
| Llama F16 | 16 | 955.7 → 1112.3 | 257.2 → 855.4 | 2.005 → 0.888 |
| Llama F16 | 128 | 5032.4 → 2792.1 | 261.9 → 856.2 | 0.644 → 0.639 |
| Llama Q8_0 | 16 | 1078.1 → 1068.0 | 301.5 → 295.7 | 1.742 → 1.764 |
| Llama Q8_0 | 128 | 9369.3 → 9271.7 | 306.6 → 297.1 | 0.373 → 0.368 |
| Qwen F16 | 16 | 577.1 → 700.2 | 132.7 → 433.4 | 3.451 → 1.634 |
| Qwen F16 | 128 | 3361.8 → 2327.4 | 141.4 → 437.0 | 0.924 → 0.901 |
| Qwen Q8_0 | 16 | 588.2 → 576.9 | 163.4 → 158.2 | 3.079 → 3.161 |
| Qwen Q8_0 | 128 | 5318.6 → 5187.1 | 172.2 → 160.9 | 0.625 → 0.642 |

[Raw model/projection/attention results](benchmark-runs/2026-10-15-parallel-single-native-thread.csv) include two workers, Lisp allocations, F32/BF16 and packed projections, and forced encoded/tiled half-weight paths. The [native-default synthetic run](benchmark-runs/2026-10-15-parallel-native-default.csv) records the native-threading comparison separately. Small Q8 differences do not establish a general scheduling threshold.

## Half-weight dispatch profile

Automatic F16/BF16 dispatch uses encoded matmul for one activation row with directly readable weight rows on the measured M1/SBCL profile. Offsets and padding qualify; irregular weight layouts and larger executor-bound batches use tiles. The caller-owned executor is validated but not invoked for the single-row projection.

The [raw dispatch results](benchmark-runs/2026-10-15-parallel-half-dispatch.csv) compare `baseline-auto`, current `auto` and forced `encoded`/`tiled` paths. Three fresh processes measure projections for each native-thread setting, with serial execution and reusable two-/four-worker pools. Complete model measurements use one native thread; native-default measurements cover projections. [Source and build metadata](benchmark-runs/2026-10-15-parallel-half-dispatch.json) identify the measured inputs and local SIMD source snapshot.[^half-build][^half-profile]

All 64 measured executor-bound single-row mapped/padded groups exceed 10% savings against forced tiled execution; median speedups range from **3.58× to 5.67×** across both native-thread settings. Larger batches and irregular views retain tiled automatic dispatch.

Complete model medians below compare **baseline-auto → auto** with the same executor. Generation includes prefill and four checked tokens, with one native thread. Native-default evidence covers projections only. These gains compare dispatch policies, not worker counts.

| Model | Prompt tokens | Workers | Decode, ms | Generation, tokens/s |
|---|---:|---:|---:|---:|
| Llama F16 | 16 | 2 | 861.2 → 252.3 | 0.892 → 2.013 |
| Llama F16 | 16 | 4 | 865.4 → 253.4 | 0.882 → 1.967 |
| Llama F16 | 128 | 2 | 856.8 → 255.6 | 0.564 → 0.875 |
| Llama F16 | 128 | 4 | 862.0 → 254.9 | 0.633 → 1.055 |
| Qwen F16 | 16 | 2 | 433.9 → 130.5 | 1.655 → 3.449 |
| Qwen F16 | 16 | 4 | 431.5 → 128.2 | 1.625 → 3.373 |
| Qwen F16 | 128 | 2 | 439.0 → 134.9 | 0.830 → 1.125 |
| Qwen F16 | 128 | 4 | 435.9 → 130.9 | 0.902 → 1.262 |

For Llama F16 decode after 16 prompt tokens with two workers and one native thread, Lisp allocation falls from 344.4 to 11.9 MiB per call. Q8 controls and serial comparisons remain in the raw results; they do not establish a universal worker-count threshold.

```sh
python3 tools/parallel-profile.py /tmp/parallel-half-dispatch.csv
```

The Python runner requests complete model/projection grids for both native-thread settings. For a projection-only run, set `CL_INFERENCE_PARALLEL_HALF_ONLY=1` when invoking `tools/parallel-bench.lisp`; unset `VECLIB_MAXIMUM_THREADS` for native-default measurements. `CL_INFERENCE_PARALLEL_MODELS_ONLY=1` selects real models; `CL_INFERENCE_PARALLEL_MODELS=f16,qwen-f16` and `CL_INFERENCE_PARALLEL_PATHS=baseline-auto,auto` narrow that run.

## Acceptance

Strict serial and parallel real-model validation uses the existing CPU oracle bounds and requires all inputs:

```sh
CL_INFERENCE_QUANT_ACCEPTANCE_STRICT=1 \
  sbcl --dynamic-space-size 10GB --script tools/parallel-acceptance.lisp
```

## Limitations

- Standard tensor matmul schedules output rows; custom packed kernels own their scheduling. One-position decode projections can expose one row; output-feature splitting is outside this interface.
- Individual contractions remain serial: [tensor contraction scheduling](https://github.com/communal-software/cl-tensor/issues/18).
- Additional workers can cost more than they save, especially for small operations and tiled half-precision decode. There is no universal worker-count recommendation.
- The automatic executor-bound half-precision profile covers one activation row and directly readable weight rows on M1/SBCL; larger batches and irregular weight layouts use tiles: [parallel dispatch measurements](https://github.com/communal-software/cl-inference/issues/55).
- Executor safety depends on the caller honoring the synchronous range contract. Sessions and mutable inputs must not be concurrently modified.

[^tensor]: Tensor matmul and reduction scheduling follow the upstream [executor contract](https://github.com/communal-software/cl-tensor/blob/trunk/docs/execution.md). Foreign storage remains caller-owned; tensor operations establish their worker storage contexts.

[^measurement]: Projection cases report three calibrated batches after warmup; other cases use one warmup and three measured calls with full collection before each sample. Decode preparation is untimed; generation resets its session and requires four expected greedy tokens. Timed calls include their own allocation and collection costs. SBCL Lisp allocation excludes native allocations and has accounting granularity. See the [dispatch methodology](#half-weight-dispatch-profile) for the repeated half-weight profile.

[^profile]: 15 October 2026; macOS 15.7.5 (24G624), Apple M1 with 8 GB RAM, SBCL 2.6.8, native SIMD and a 3 GB Lisp heap. Models use the existing pinned CPU acceptance artifacts. cl-tensor: `81272fa7ae765673ddaf43e0f984a1b035274ff4`; trivial-simd: `f5d8ad5b70deecc16df5cad5dde1f8de959b4104`; native library SHA-256: `32a032f959e0122a74290f7af9a83edfe026ce63cb9bf317e305d9534fb0faa6`. Reference timings are three-sample medians from one process per native-threading setting; each decode sample rebuilds its prefix before full collection and timing. They describe this profile, not other hardware or pools.

[^half-profile]: Projection calls include output allocation and any layout/conversion scratch. Each process reports three calibrated batches of at least 10 ms after warmup and an initial full collection. Model cases have one warmup and three measured calls, with full collection before each sample. Decode reuses one prepared prefix and overwrites the same final position; generation resets its session before each sample and requires four expected greedy tokens. Forced paths cover projections and full-model decode; prefill uses automatic dispatch, and generation compares automatic dispatch with `baseline-auto`. Summaries use medians across the three processes; model summaries cover the single-native-thread setting. The runner rejects unrelated processes using at least 50% CPU in three consecutive snapshots, checked before and after each process and during output after 60 seconds. Interrupted native-default model processes are excluded after repeated macOS background CPU interference; only the complete projection sweeps are retained for that setting. SBCL Lisp allocation excludes native memory and has accounting granularity; zero does not establish allocation-free execution.

[^half-build]: The measured SIMD checkout contains pre-existing local strided/BLAS changes. Metadata record its runtime/build source hashes alongside the native-library hash; the Git revision alone does not identify this snapshot. Its encoded, Q8 and packed-matmul Lisp sources match published [runtime reference `cd61d43`](https://github.com/communal-software/trivial-simd/commit/cd61d43c2986c05ac32503d7619b2e7b67d1b584). Other builds need independent measurements.
