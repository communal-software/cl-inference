# Safetensors

`cl-inference/safetensors` reads local Safetensors files and HF model directories through the [weights protocol](weights.md). It registers `:safetensors` and exposes mapped tensor views plus optional sibling `config.json` metadata.

```lisp
(asdf:load-system :cl-inference/safetensors)
(local:with-weights (w #p"models/Qwen3-0.6B/")
  (list (local:weights-metadata w "model_type")
        (local:weights-tensor-names w)
        (local:weights-tensor w "model.embed_tokens.weight")))
```

## Inputs

| Input | Reads |
|---|---|
| A `.safetensors` file | That file and optional sibling `config.json` |
| `model.safetensors.index.json` | Every shard named by `weight_map`, plus optional sibling configuration |
| A model directory | The index when present; otherwise `model.safetensors` |

An index takes precedence over a single model file. Directories without either conventional filename are rejected; the reader does not guess from a glob. Shards must resolve inside the model directory, including through symlinks. Tensor names follow header declaration order within lexically ordered shard paths.

Files are detected by their extension or plausible header prefix; indexes use their conventional filename. Explicit `:format :safetensors` skips detection. All opens validate the full header and payload.

## Configuration

Configuration keys keep their HF names. Nested objects are hash tables, arrays are vectors, booleans are `t`/`nil`, and JSON null is `cl:null`. Missing keys return the supplied default and NIL presence; a present null returns `cl:null` and T.

```lisp
(local:weights-metadata w "hidden_size")
(gethash "bits" (local:weights-metadata w "quantization"))
```

`safetensors.__metadata__` holds a hash table from relative shard filenames to their header `__metadata__` objects. It is separate from configuration keys; `config.json` cannot use that reserved key. Header metadata values must be strings.

Configuration is optional for inspection. `weights-architecture-name` returns the string `model_type`, or NIL when absent; `load-architecture` requires a matching architecture and its required hparams.

## Tensor types and lifetime

| File dtype | Typed view |
|---|---|
| F32, F64, F16, BF16 | `:f32`, `:f64`, `:f16`, `:bf16` |
| I8/I16/I32/I64 | `:s8`, `:s16`, `:s32`, `:s64` |
| U8/U16/U32/U64 | `:u8`, `:u16`, `:u32`, `:u64` |
| BOOL, F8_E4M3, F8_E5M2, F8_E8M0 | Raw bytes only; typed access signals `unsupported-weights-type` |

Tensor info retains the file dtype as a keyword, such as `:i32`. Shapes are row-major and are not reversed. Scalars and empty tensors are supported. Unknown and sub-byte encodings are rejected.

Each file is mapped once, copy-on-write. Views share its payload; writes do not change the source file. Views are invalid after `close-weights`. Closing is idempotent and attempts to release every shard, even if an unmap fails.[^mapping]

## Model loading

The shipped Llama, Qwen3 and Gemma 3 architectures map HF tensor names and configuration fields in their `:safetensors` sections. All use split-half RoPE; Qwen3 adds Q/K normalization. `head_dim` overrides `hidden_size / num_attention_heads`, and missing Llama/Qwen3 output weights require `tie_word_embeddings=true`. Gemma uses the architecture default of tied embeddings when that field is absent.

```lisp
(local:with-weights (w #p"models/Qwen3-0.6B/")
  (let ((model (local:load-model w)))
    (local:model-forward model '(1 2 3))))
```

CPU fixtures verify F32/F16/BF16 logits and cached decoding for all three architectures. [Gemma 3 270M IT](gemma-hf.md) also has independent real-model F32 acceptance, local/global attention and HF-only norm offsets. Ordinary RoPE and `rope_scaling` with `rope_type="llama3"` are supported.[^rope] Unsupported architecture-specific scaling, bias, activation and sliding-window variants signal `architecture-error` during model loading.

Packed MLX integers, scales, biases and quantization metadata remain available through the weights protocol. CPU model loading rejects quantized configurations. The [HF tokenizer reader](hf-tokenizers.md) supplies Llama/Qwen3/Gemma chat setup. The [MLX backend](mlx.md) executes packed affine 4-bit models. Its [real-model acceptance](mlx-acceptance.md) verifies Llama 3.2 1B Instruct and Qwen3 0.6B in BF16 and affine 4-bit form.

## Validation

Malformed files signal `safetensors-error`, a `weights-error` subclass. Validation covers strict UTF-8 JSON, duplicate keys, headers over 100 MB, nesting over 128 levels, invalid shapes/dtypes, payload-size mismatches, gaps, overlaps, truncation, index ownership and duplicate tensors across shards. Failed opens release acquired mappings.

## Limitations

| Missing | Tracker |
|---|---|
| Broader HF pipelines and templates; additional MLX formats | [HF limitations](hf-tokenizers.md#limitations), [MLX limitations](mlx.md#limitations) |
| Additional RoPE modes and encodings | Outside the documented coverage |
| Windows mapping; BSD verification | [#20](https://github.com/communal-software/cl-inference/issues/20), [#21](https://github.com/communal-software/cl-inference/issues/21) |

[^mapping]: The optional internal `cl-inference/mapping` system shares POSIX `mmap` helpers with GGUF. Zero-copy typed access assumes a little-endian host.
[^rope]: Llama 3 frequency scaling follows [Transformers' reference implementation](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/modeling_rope_utils.py). Header layout and payload validation follow the [Safetensors specification](https://github.com/safetensors/safetensors/blob/main/README.md).
