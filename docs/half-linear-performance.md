# Half-precision linear performance

F16 and BF16 linear weights use direct encoded matmul for small activation batches on the measured configuration. Results are F32. Larger batches and other configurations use bounded tiled BLAS.

## Dispatch profile

| Encoding | Direct encoded activation rows | Larger batches |
| --- | ---: | --- |
| F16 | 1–18 | Tiled BLAS |
| BF16 | 1–21 | Tiled BLAS |

The profile requires macOS, Apple M1 ARM64, SBCL 2.6.8, the native SIMD backend with encoded matmul available, and no tensor executor. Compatible mapped weights retain their storage and offsets; see [half-precision linear](blocks.md#half-precision).

## Measurements

Whole `linear` calls include output allocation and any conversion scratch. For an 8192 × 2048 weight matrix:

| Encoding | Activation rows | Encoded ms | Tiled ms |
| --- | ---: | ---: | ---: |
| F16 | 1 | 3.207 | 11.032 |
| F16 | 6 | 4.694 | 11.292 |
| F16 | 46 | 33.905 | 29.333 |
| BF16 | 1 | 3.083 | 11.847 |
| BF16 | 6 | 4.712 | 11.929 |
| BF16 | 46 | 33.738 | 29.857 |

Each cutoff is the largest contiguous prefix starting at one row with at least 10% savings for every measured shape. Five fresh processes measure counts 1–24 and 46; the increasing sweep stops after both encodings first fail the rule. Each process reports the median of three calibrated batches lasting at least 10 ms, after warmup.[^inputs]

[Raw timings and Lisp allocations](benchmark-runs/2026-10-14-half-linear.csv) include every process and shape. A separate [automatic dispatch run](benchmark-runs/2026-10-14-half-linear-auto.csv) checks counts 1, 6, 18, 19, 21, 22 and 46. Allocation figures use SBCL bytes consed; they exclude native memory and have accounting granularity, so zero does not establish allocation-free execution.

## Reproduce

Run benchmarks separately from tests and other CPU-intensive work:

```sh
python3 tools/half-linear-profile.py /tmp/half-linear-profile.csv
```

For a single process or automatic dispatch verification:

```sh
CL_INFERENCE_HALF_BENCH_COUNTS=1,6,46 \
  sbcl --dynamic-space-size 4096 --script tools/half-linear-bench.lisp
CL_INFERENCE_HALF_BENCH_COUNTS=1,6,18,19,21,22,46 \
  CL_INFERENCE_HALF_BENCH_AUTO=1 \
  sbcl --dynamic-space-size 4096 --script tools/half-linear-bench.lisp
```

Set `CL_INFERENCE_HALF_BENCH_FULL_SWEEP=1` for all counts through 64. The benchmark requires approximately 1 GB of temporary mapped weight files.[^build]

## Limitations

This profile covers one hardware/runtime configuration and the listed model shapes. It does not establish a speedup for every shape. Tile boundaries make performance nonmonotonic; larger counts can occasionally benefit from direct dispatch without satisfying the contiguous-prefix rule.

Other machines, runtimes and executor configurations need independent whole-call measurements. Larger-batch native kernels and x86 tuning are tracked in [trivial-simd issue 74](https://github.com/communal-software/trivial-simd/issues/74).

[^inputs]: Synthetic mapped weights alternate 1 and −0.5, with activations 0.5 and checked F32 outputs. Mapping initialization is outside timing. Weight shapes are 8192×2048, 2048×8192, 512×2048, 2048×2048, 128256×2048, 3072×1024, 1024×3072, 1024×1024, 1024×2048, 2048×1024 and 151936×1024, covering the Llama and Qwen projection/head dimensions.

[^build]: Measured on macOS 15.7.5 (24G624), Apple M1, SBCL 2.6.8, native backend, no executor; cl-tensor `9504a3130554c0ccf82281eb78dd1f05dccc383b` and trivial-simd `04f5ff4fbf1475ef3d60c227384f685896d8b89f`. Native Release build uses AppleClang 17, `-O3 -DNDEBUG` and `-ffp-contract=off`; library SHA256 is `ec434475e8e60e3e254dc0935b3e877e0ce8be620cd29ee6332f57e1b21c51bb`.
