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

Set `CL_INFERENCE_PARALLEL_SYNTHETIC_ONLY=1` to omit real models. Compare native threading settings in separate processes. [Half-precision dispatch](half-linear-performance.md) retains its measured serial profile; an executor selects the conservative tiled path under automatic dispatch.

## Measured profile

On the measured M1/SBCL configuration, four workers reduce synthetic attention time by about 1.6–2.1× and improve the longer F16 prefills. End-to-end generation does not inherit those gains: automatic half-weight decode uses tiled conversion, while packed Q8 kernels retain their own scheduling. Keep serial execution as the default; choose an executor from measurements of the intended format and phase.[^profile]

The model table uses a single native BLAS thread. Each cell shows **serial → four workers**. Generation includes prefill and four completed tokens.

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
- Automatic half-precision dispatch retains its serial-only profile; executor-bound decode uses tiled conversion: [parallel dispatch measurements](https://github.com/communal-software/cl-inference/issues/55).
- Executor safety depends on the caller honoring the synchronous range contract. Sessions and mutable inputs must not be concurrently modified.

[^tensor]: Tensor matmul and reduction scheduling follow the upstream [executor contract](https://github.com/communal-software/cl-tensor/blob/trunk/docs/execution.md). Foreign storage remains caller-owned; tensor operations establish their worker storage contexts.

[^measurement]: Each case has one warm-up and three measured calls; the report uses medians. Full garbage collection precedes each sample and follows untimed decode preparation. Timed calls include their own allocation and collection costs. SBCL Lisp allocation excludes native allocations and has accounting granularity. Generation measurements require four completed tokens.

[^profile]: 15 October 2026; macOS 15.7.5 (24G624), Apple M1 with 8 GB RAM, SBCL 2.6.8, native SIMD and a 3 GB Lisp heap. Models use the existing pinned CPU acceptance artifacts. cl-tensor: `81272fa7ae765673ddaf43e0f984a1b035274ff4`; trivial-simd: `f5d8ad5b70deecc16df5cad5dde1f8de959b4104`; native library SHA-256: `32a032f959e0122a74290f7af9a83edfe026ce63cb9bf317e305d9534fb0faa6`. Timings are three-sample medians from one process per native-threading setting; they describe this profile, not other hardware or pools.
