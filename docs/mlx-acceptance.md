# MLX acceptance

The MLX acceptance suite compares Qwen3 0.6B BF16 and affine 4-bit with Python MLX using identical weights and F32 computation. Every case must run; skipped cases do not establish acceptance.

## Reference inputs

| Dependency | Pin |
|---|---|
| Qwen/Qwen3-0.6B | HF revision `c1899de289a04d12100db370d81485cdf75e47ca` |
| MLX | `0.32.2` |
| MLX-LM | `0.32.0` |
| Transformers / Tokenizers | `5.19.0` / `0.23.3` |

Models and oracle outputs stay in gitignored `reference/mlx/`. The development-only Python packages do not belong to the Lisp runtime.

## Prepare and run

```sh
python3 -m venv reference/mlx/venv
reference/mlx/venv/bin/pip install -r tools/mlx-reference-requirements.txt
reference/mlx/venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-0.6B',
    revision='c1899de289a04d12100db370d81485cdf75e47ca',
    local_dir='reference/mlx/qwen-float',
    allow_patterns=['*.json', '*.safetensors'])
PY
reference/mlx/venv/bin/python -m mlx_lm convert \
  --hf-path reference/mlx/qwen-float --mlx-path reference/mlx/qwen-4bit \
  --quantize --q-bits 4 --q-group-size 64
for kind in float 4bit; do
  reference/mlx/venv/bin/python tools/mlx-reference.py \
    "reference/mlx/qwen-$kind" "reference/mlx/qwen-$kind-oracle.json"
done
sbcl --dynamic-space-size 6GB --non-interactive --eval '(require :asdf)' \
  --eval '(asdf:test-system "cl-inference/mlx/acceptance")'
```

The converter requires an absent output directory. Preserve an existing conversion under another name before rerunning it.

## Checks and bounds

| Check | Requirement |
|---|---|
| Rendered prompt and token IDs | Exact |
| Unicode and special-token encodings | Exact |
| Final logits after chunked prefill | Maximum absolute error below `0.005`; relative L2 below `0.0005` |
| Eight cached greedy decode steps | Exact token IDs |

Verified maximum absolute errors are below `0.00007` for both variants. Relative L2 is below `0.000003`.[^precision]

The offline `cl-inference/mlx/tests` suite also covers Llama and Qwen3 F32/F16/BF16, affine 4-bit, mixed weights, quantized embeddings, tied output heads, Llama 3 RoPE, cache lifetime, client events, tools and reasoning.

## Limitations

| Deferred verification | Tracker |
|---|---|
| Real Llama models | [#51](https://github.com/communal-software/cl-inference/issues/51) |
| Additional quantization formats | [#50](https://github.com/communal-software/cl-inference/issues/50) |

[^precision]: The oracle promotes floating weights, scales and biases to F32 with `model.set_dtype(mx.float32)`; packed integer weights remain unchanged. This matches the Lisp backend's execution precision instead of comparing it with BF16 intermediate arithmetic.
