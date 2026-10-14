# Support

Local and remote models use the same [client contract](client.md): `complete` accepts neutral messages and returns `(:ok reply)` or `(:error reason)`. Choose a backend to choose where the model runs.

## Remote protocols and providers

| Interface | Registered name | Configuration |
|---|---|---|
| OpenAI-compatible chat completions | `:protocol-openai` | Base URL, model ID and headers; optional SSE streaming |
| Ollama native chat | `:protocol-ollama` | Base URL and model ID; optional NDJSON streaming |
| Built-in Ollama provider | `:ollama` | Native protocol, default `http://127.0.0.1:11434`, no authentication |
| Custom provider | Name passed to `define-provider` | Either protocol, with endpoint, authentication, defaults and optional rewrites |

The server determines available models and their capabilities. Remote models do not need a local architecture or weight loader. Ollama's advertised model list is not a restriction; choose a model installed on the server.

HTTP-only clients load `cl-inference` without the local engine or cl-tensor. Client tests pass on SBCL, CCL and ECL.

## Local architectures

| Architecture | Verified real model | GGUF storage | Tokenizer and template |
|---|---|---|---|
| `llama` | Llama 3.2 1B Instruct | F16, Q8_0 | `llama-bpe`, `llama3-chat` |
| `qwen3` | Qwen3 0.6B | F16, Q8_0 | `qwen2`, `qwen3-chat` |

Local inference is verified on SBCL. The architecture declarations cover these families; the acceptance suite establishes support for the listed model variants. Other sizes and variants require their own verification.

`define-architecture` describes the local engine's blocks and weight mappings. Remote providers are declared separately with `define-provider`. See [Architectures](architectures.md), [Tokenizers](tokenizers.md) and [Chat templates](chat-templates.md) for extension points.

## Local formats and storage

| Support | System | Coverage |
|---|---|---|
| GGUF v3, little-endian | `cl-inference/gguf` | Model metadata and mapped tensor views |
| F32, F16, BF16 inference weights | `cl-inference/local` and the format loader | Offline model/block fixtures; F16 also has real-model acceptance |
| Q8_0 inference weights | `cl-inference/quant` | Quantized dtype and matmul; real-model acceptance |

Reading a GGUF file does not establish inference support for every tensor encoding. The reader exposes additional scalar tensor types and raw bytes for other quantized types; see [GGUF](gguf.md#tensor-types). Other quantizations and formats are [limitations](#limitations).

## Choose a backend

```lisp
(defun ask (backend)
  (cl-inference:complete backend
                        :messages '((:role :user :content "Hello!"))
                        :temperature 0 :max-tokens 64))

(ask (cl-inference:make-provider :ollama
                                :base-url "http://model-server:11434"
                                :model "your-model-id"))

(cl-inference/local:with-weights (weights "model.gguf")
  (ask (cl-inference/local:make-local-backend weights)))
```

Load `cl-inference/gguf` for a GGUF model and `cl-inference/quant` for Q8_0. The local backend borrows weights; keep them open throughout every completion. Its default serialized session reuses one cache; `:session-policy :per-call` permits concurrent calls with separate caches. See [Local backend](local-backend.md) for options.

## Limitations

| Missing | Tracker |
|---|---|
| Local ONNX models; execution approach remains undecided | [#47](https://github.com/communal-software/cl-inference/issues/47) |
| Local Gemma-family models; initial version and modalities remain undecided | [#48](https://github.com/communal-software/cl-inference/issues/48) |
| Safetensors and HF config.json loader | [#13](https://github.com/communal-software/cl-inference/issues/13) |
| Q4_0, Q4_K and Q6_K inference | [#14](https://github.com/communal-software/cl-inference/issues/14) |
| GGUF tensor loading on CCL and ECL | [#45](https://github.com/communal-software/cl-inference/issues/45) |
| Identical seeded sampling on CCL | [#46](https://github.com/communal-software/cl-inference/issues/46) |

Other engine and tokenizer gaps are listed in [Limitations](limitations.md).
