# Limitations

| Missing | Tracker |
|---|---|
| Local ONNX models | [#47](https://github.com/communal-software/cl-inference/issues/47) |
| Local Gemma-family models | [#48](https://github.com/communal-software/cl-inference/issues/48) |
| HF tokenizer files, template detection and MLX quantized execution | [#43](https://github.com/communal-software/cl-inference/issues/43) |
| GGUF tensor fields are lost on CCL and ECL; use SBCL for local inference | [#45](https://github.com/communal-software/cl-inference/issues/45) |
| Seeded sampler sequence differs on CCL | [#46](https://github.com/communal-software/cl-inference/issues/46) |
| Windows and BSD file mapping | [#20](https://github.com/communal-software/cl-inference/issues/20), [#21](https://github.com/communal-software/cl-inference/issues/21) |
| Removing an inherited architecture entry | [#23](https://github.com/communal-software/cl-inference/issues/23) |
| Q8_0 matmul over many activation rows is slower than dequantized tiles through BLAS | [#41](https://github.com/communal-software/cl-inference/issues/41) |
| Adopt upstream half-precision matmul; `linear` converts tiles to f32 first | [#42](https://github.com/communal-software/cl-inference/issues/42) |
| Chat templates: special-token text in messages, more families | [#37](https://github.com/communal-software/cl-inference/issues/37), [#38](https://github.com/communal-software/cl-inference/issues/38) |
| Tokenizer gaps: files without `pre`, other `pre` types, SentencePiece, rule placement | [#32](https://github.com/communal-software/cl-inference/issues/32), [#33](https://github.com/communal-software/cl-inference/issues/33), [#34](https://github.com/communal-software/cl-inference/issues/34) |
| Sampler sorts the whole vocabulary each token | [#40](https://github.com/communal-software/cl-inference/issues/40) |
| Reusing the KV cache across turns; slow cache copy | [#16](https://github.com/communal-software/cl-inference/issues/16), [#39](https://github.com/communal-software/cl-inference/issues/39) |

The full roadmap is [#1](https://github.com/communal-software/cl-inference/issues/1).
