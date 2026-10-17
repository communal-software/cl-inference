# Limitations

| Missing | Tracker |
|---|---|
| ONNX exports and capabilities outside SmolLM2 FP32 CPU text chat | [ONNX limitations](onnx.md#limitations) |
| Gemma formats, families and modalities outside Gemma 3 270M IT Q8_0 CPU | [Gemma limitations](gemma.md#limitations) |
| Broader HF pipelines and templates; additional MLX formats | [HF limitations](hf-tokenizers.md#limitations), [MLX limitations](mlx.md#limitations) |
| Windows and BSD file mapping | [#20](https://github.com/communal-software/cl-inference/issues/20), [#21](https://github.com/communal-software/cl-inference/issues/21) |
| Q8_0 matmul over many activation rows is slower than dequantized tiles through BLAS | [#41](https://github.com/communal-software/cl-inference/issues/41) |
| Half-precision direct dispatch has a measured M1/SBCL profile; other configurations use tiled BLAS | [Half-precision limitations](half-linear-performance.md#limitations) |
| Templates outside pinned text coverage; semantic prompt injection (see [template limits](chat-families.md#limitations)) | [#37](https://github.com/communal-software/cl-inference/issues/37), [#38](https://github.com/communal-software/cl-inference/issues/38) |
| Other GGUF BPE `pre` types and SentencePiece variants | [#61](https://github.com/communal-software/cl-inference/issues/61), [#62](https://github.com/communal-software/cl-inference/issues/62) |
| SentencePiece automatic EOS insertion and extra-whitespace normalization | [#65](https://github.com/communal-software/cl-inference/issues/65), [#66](https://github.com/communal-software/cl-inference/issues/66) |
| Sampler sorts the whole vocabulary each token | [#40](https://github.com/communal-software/cl-inference/issues/40) |
| Slow cache copy | [#39](https://github.com/communal-software/cl-inference/issues/39) |
| Q8 conversion speed and full-prefix attention scratch | [#57](https://github.com/communal-software/cl-inference/issues/57) |

The [cross-repository roadmap](ROADMAP.md) tracks priorities and next actions; [#1](https://github.com/communal-software/cl-inference/issues/1) coordinates the parent project.

[Packed GGUF on MLX](mlx-gguf.md) supports Q4_0, Q4_K and Q6_K weights. Its [limitations](mlx-gguf.md#limitations) cover other weight encodings and platform support.
