# MLX GGUF measurements

The profile measures complete model loading, session prefill and eight cached decode calls on Apple Silicon. Packed weights retain their original byte encoding throughout every trial.

## Reproduce

Prepare the six pinned models using [GGUF acceptance](mlx-gguf-acceptance.md), then run without other inference or reference-generation processes:

```sh
python3 tools/mlx-gguf-profile.py reference/mlx-gguf-profile --tokens 64 --batch 64
```

Each model runs in a fresh SBCL process. Trial zero records cold kernel compilation and warmup separately; trials one through three measure steady-state calls. Prefill consumes 64 token IDs with a batch size of 64; each decode call appends one token. The F32 cache holds 80 positions. Timings include lazy graph construction, evaluation, cache updates and final-logit readback. They exclude CPU sampling and tokenization.[^memory]

The output includes per-model CSV rows and logs, plus metadata with hardware, runtime, dependency revisions, model hashes and process peak RSS. `packed_bytes` is recomputed from MLX byte arrays after every measured phase and must match their logical block dimensions.

## Recorded profile

Apple M1, 8 GiB RAM, SBCL 2.6.8, macOS; recorded 2026-10-16. [Metadata](benchmark-runs/2026-10-16-mlx-gguf.json) records exact platform, dependency revisions and model hashes. Times are seconds; steady-state values are medians of three trials. Peaks include cold and steady-state phases.

| Model | Load | Prefill 64 | Decode 8 | MLX peak MiB | Process peak MiB |
|---|---:|---:|---:|---:|---:|
| llama-q4_0 | 1.703614 | 1.682 | 0.480 | 976.8 | 2031.5 |
| llama-q4_k | 3.722942 | 3.940 | 0.855 | 1011.9 | 2107.9 |
| llama-q6_k | 4.208063 | 2.864 | 0.704 | 1216.3 | 2511.6 |
| qwen-q4_0 | 1.678353 | 0.814 | 0.306 | 708.5 | 1459.8 |
| qwen-q4_k | 1.811192 | 1.839 | 0.471 | 723.2 | 1488.9 |
| qwen-q6_k | 1.904691 | 1.349 | 0.380 | 856.2 | 1754.3 |

Raw CSVs use the `2026-10-16-mlx-gguf-<model>.csv` prefix in [benchmark-runs](benchmark-runs/). They retain cold prefill/decode, each steady-state trial, active allocations and packed storage sizes.

## Limitations

Prefill streams packed rows independently for each token. Sharing packed tiles across token rows is tracked by [#68](https://github.com/communal-software/cl-inference/issues/68).

These measurements describe one machine and prompt length. They establish packed storage and bounded dequantization memory; they do not establish a speedup over another backend or predict long-context performance. GPU sampling, attention fusion and cache-update tuning are separate work.

[^memory]: MLX active and peak allocations come from the native MLX C API. The allocator cache is cleared before trial zero and retained during steady-state trials; peak allocation counters reset before each measured phase. Weight copies remain resident. Model loading includes opening/mapping the file, metadata validation, finite-scale scanning, byte import and initial Metal initialization. Process peak RSS includes Lisp, mapped pages and native resources; it is separate from MLX allocation totals. Projection scratch is 32 F32 values (128 bytes) per workgroup, with decoded values in registers; no floating-point weight tile or full-matrix buffer is allocated.
