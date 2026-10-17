# ONNX acceptance

Strict acceptance verifies the pinned SmolLM2 135M FP32 graph against independent Python ONNX Runtime and Hugging Face tokenization. Skipped checks do not establish model support.

## Prepare

Install/build [cl-onnx](onnx.md#setup-and-loading), then:

```sh
uv run --with-requirements tools/onnx-reference-requirements.txt python tools/onnx-reference.py --download
CL_INFERENCE_ONNX_ACCEPTANCE_STRICT=1 sbcl --non-interactive \
  --eval '(require :asdf)' --eval '(asdf:test-system "cl-inference/onnx/acceptance")'
```

The preparation tool downloads missing artifacts, verifies every committed SHA256 and generates the oracle in gitignored `reference/onnx/`. Existing files with mismatched hashes fail. Running without `--download` verifies existing assets and regenerates the reference without downloading model files.

## Pinned inputs

| Input | Pin |
|---|---|
| Model | `onnx-community/SmolLM2-135M-Instruct-ONNX` |
| Revision | `b8a5c0f183b78c55955a5364f610c36668b5e681` |
| Graph | `onnx/model.onnx`, FP32, approximately 539 MB |
| Graph IR/opset | IR 7, default ONNX opset 14 |
| Native and Python runtime | ONNX Runtime 1.31.0, CPU, one intra-op thread |
| Artifact hashes | `tests/onnx/model.json` |
| Oracle hash | `tests/onnx/oracle.sha256` |
| Python dependencies | `tools/onnx-reference-requirements.txt` |

## Checks

Three chats cover default and explicit system messages, multiple turns, Unicode and arithmetic. Tests require exact rendered prompt IDs and greedy generated IDs, compare full initial and cached-decode logits with `atol=1e-4`, `rtol=1e-4`, and compare neutral streamed text and the closing event. Prefill batching exercises K/V continuation. Tokenizer cases include every ASCII control byte, missing byte tokens, digits, Unicode, whitespace and special tokens.

Acceptance checks the manifest, model sidecars, graph, runtime and oracle hashes. Missing oracle data skips ordinary runs and fails strict runs. Native and backend offline fixtures run without these large reference assets:

```lisp
(asdf:test-system "cl-onnx")
(asdf:test-system "cl-inference/onnx")
```

Regenerate the tiny backend fixtures with `uv run --with onnx==1.20.1 python tools/onnx-fixtures.py`. They exercise native cancellation and timeout, queue waiting, recovery, stops, sampling, resource ownership and malformed packages.

## Limitations

| Unverified coverage | Tracker |
|---|---|
| Other model/export, precision and execution-provider combinations | [#72](https://github.com/communal-software/cl-inference/issues/72) |
