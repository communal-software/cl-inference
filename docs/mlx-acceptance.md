# MLX acceptance

The MLX acceptance suite compares Llama 3.2 1B Instruct and Qwen3 0.6B in original BF16 and affine 4-bit form with Python MLX. Both sides use identical weights and F32 computation. Strict acceptance requires all four cases to pass without skips.

[Gemma HF acceptance](gemma-hf.md#acceptance) uses an independent Transformers F32 oracle and has a separate strict flag and suite.

[Cache acceptance](cache-acceptance.md) adds prefix replay and Q8 KV checks for all four weight variants. Set its strict environment variable as well when validating caches.

## Reference inputs

| Dependency | Pin |
|---|---|
| meta-llama/Llama-3.2-1B-Instruct | HF revision `9213176726f574b556790deb65791e0c5aa438b6` |
| Qwen/Qwen3-0.6B | HF revision `c1899de289a04d12100db370d81485cdf75e47ca` |
| MLX | `0.32.2` |
| MLX-LM | `0.32.0` |
| Transformers / Tokenizers | `5.19.0` / `0.23.3` |
| 4-bit conversion | Affine, group size `64` |

Models and oracle outputs stay in gitignored `reference/mlx/`. The committed `tools/mlx-reference-manifest.json` pins their SHA-256 hashes and oracle provenance. Hash mismatches fail even when other inputs are missing. The Python packages are development dependencies.

## Prepare

Use macOS Apple Silicon with Metal access. Llama downloads require approved access to the official HF repository and an authenticated HF account.

```sh
python3 -m venv reference/mlx/venv
reference/mlx/venv/bin/pip install -r tools/mlx-reference-requirements.txt
reference/mlx/venv/bin/hf auth login
reference/mlx/venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
models = [
    ('llama', 'meta-llama/Llama-3.2-1B-Instruct',
     '9213176726f574b556790deb65791e0c5aa438b6'),
    ('qwen', 'Qwen/Qwen3-0.6B',
     'c1899de289a04d12100db370d81485cdf75e47ca'),
]
for name, repository, revision in models:
    snapshot_download(repository, revision=revision,
        local_dir=f'reference/mlx/{name}-float',
        allow_patterns=['config.json', 'generation_config.json',
                        'special_tokens_map.json', 'tokenizer.json',
                        'tokenizer_config.json', 'model*.safetensors',
                        'model.safetensors.index.json'])
PY
for model in llama qwen; do
  reference/mlx/venv/bin/python -m mlx_lm convert \
    --hf-path "reference/mlx/$model-float" \
    --mlx-path "reference/mlx/$model-4bit" \
    --quantize --q-bits 4 --q-group-size 64
  for kind in float 4bit; do
    reference/mlx/venv/bin/python tools/mlx-reference.py \
      "reference/mlx/$model-$kind" "reference/mlx/$model-$kind-oracle.json"
  done
done
```

Conversion requires an absent output directory. Preserve existing conversions under another name before rerunning. Single-file and indexed Safetensors inputs are supported; every consumed shard and sidecar must match the manifest.[^manifest]

## Run

```sh
CL_INFERENCE_MLX_ACCEPTANCE_STRICT=1 \
  sbcl --dynamic-space-size 6GB --non-interactive --eval '(require :asdf)' \
  --eval '(asdf:test-system "cl-inference/mlx/acceptance")'
```

Without strict mode, missing inputs skip their case. Skips do not establish acceptance. In a REPL, load `cl-inference/mlx/acceptance` and call `(cl-inference/mlx/tests:run-acceptance :strict t)`; it returns true only when all cases pass.

## Checks and bounds

| Check | Requirement |
|---|---|
| Complete rendered prompt and chat IDs | Exact; Llama date is `26 Jul 2024`, with one leading BOS |
| Unicode, combining characters, numbers and control tokens | Exact; Llama probes cover BOS enabled and disabled |
| Final logits after chunked prefill (`n-batch 7`) | Matching dimensions, finite values, maximum absolute error below `0.005`, relative L2 below `0.0005` |
| Logits after each of eight cached decode steps | Same numerical bounds |
| Eight cached greedy token IDs | Exact |
| Public `generate` after session reset | Exact IDs before the first stop token and matching finish reason |

The runner prints maximum absolute error and relative L2 for prefill and every decode step.[^precision] Verified worst errors across all nine comparisons are:

| Variant | Maximum absolute error | Relative L2 |
|---|---:|---:|
| Llama BF16 | `5.82e-5` | `4.45e-6` |
| Llama affine 4-bit | `3.77e-5` | `2.82e-6` |
| Qwen BF16 | `6.96e-5` | `3.12e-6` |
| Qwen affine 4-bit | `6.68e-5` | `3.59e-6` |

The offline `cl-inference/mlx/tests` suite covers Llama and Qwen3 F32/F16/BF16, affine 4-bit, mixed weights, quantized embeddings, tied output heads, Llama 3 RoPE, cache lifetime, client events, tools and reasoning. Acceptance-harness fixtures cover hashes, provenance, missing inputs, strict mode, indexed shards, numerical validation and stop handling.

## Limitations

| Deferred verification | Tracker |
|---|---|
| Additional quantization formats | [#50](https://github.com/communal-software/cl-inference/issues/50) |

[^manifest]: Regenerating oracles does not update trusted hashes. Maintainers use the oracle command with `--manifest tools/mlx-reference-manifest.json` to update one case, review the resulting hashes and provenance, then run strict acceptance. Dependency versions are checked before generation. File hashes are cached only within an acceptance run.

[^precision]: The oracle promotes floating weights, scales and biases to F32 with `model.set_dtype(mx.float32)`; packed integer weights remain unchanged. This matches the Lisp backend's execution precision.
