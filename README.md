# cl-inference

> **Work in progress.** This project is under development; expect missing features and breaking changes.

One `complete` contract for hosted and local LLMs in Common Lisp. `cl-inference/local` runs models in-process on [cl-tensor](https://github.com/communal-software/cl-tensor); its architectures are declared in a Lisp DSL where a similar architecture inherits from an existing one and overrides what differs.

## Support

| Area | Implemented |
|---|---|
| Remote protocols | OpenAI-compatible chat completions; Ollama native chat |
| Remote providers | Built-in `:ollama`; custom providers through `define-provider` |
| Local architectures | Llama and Qwen3; verified with Llama 3.2 1B Instruct and Qwen3 0.6B |
| Local formats | GGUF v3; Safetensors with HF configuration and indexed shards |
| Inference weights | F32, F16, BF16, Q8_0, Q4_0, Q4_K and Q6_K |

Real-model CPU inference is verified on SBCL; the HTTP client and CPU fixture suites pass on SBCL, CCL and ECL. Safetensors CPU models are fixture-tested; the optional MLX backend has Llama 3.2 1B Instruct and Qwen3 0.6B float and 4-bit acceptance. See [Support](docs/support.md) for configuration, verified variants and limitations. Local Gemma and ONNX models are planned.

## Docs

- Start here: [Getting started](docs/getting-started.md), [Runnable examples](docs/examples.md)
- Calling models: [Client](docs/client.md), [Local backend](docs/local-backend.md), [MLX](docs/mlx.md)
- Extending models: [Architectures](docs/architectures.md), [Tokenizers](docs/tokenizers.md), [Chat templates](docs/chat-templates.md), [Blocks](docs/blocks.md)
- Generation: [Sessions](docs/sessions.md), [Sampling](docs/sampling.md)
- Storage: [Weights](docs/weights.md), [GGUF](docs/gguf.md), [Safetensors](docs/safetensors.md), [Quantized weights](docs/quant.md)
- Development: [Design](docs/design.md), [Testing](docs/testing.md), [Model acceptance](docs/acceptance.md), [Limitations](docs/limitations.md)

Work is tracked at <https://github.com/communal-software/cl-inference/issues>.

## License

[GPLv3](LICENSE).
