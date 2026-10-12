# Blocks

A block is one computation of a model: a norm, an attention, an MLP. An [architecture](architectures.md) names its blocks; `load-model` creates them and binds each to its tensors.

```lisp
(local:with-weights (w "Llama-3.2-1B-Instruct-f16.gguf")
  (let ((model (local:load-model w)))
    (local:model-forward model '(128000 9906))))   ; logits of the last position, shape (1 vocab)
```

A model holds views into `w` and is valid until `w` closes.

## Standard blocks

| Block | Scope | Tensor roles | Hparams | Returns |
|---|---|---|---|---|
| `embedding` | model | `embd` | | Rows of the table for the token ids, as f32 |
| `attention` | layer | `norm q k v output`, optional `rope-freqs`, with `:qk-norm`: `q-norm k-norm` | `n-heads rms-epsilon`; `n-kv-heads` `head-dim` `rope-dims` `rope-theta` | Residual delta |
| `swiglu-mlp` | layer | `norm gate up down` | `rms-epsilon` | Residual delta |
| `lm-head` | model | `norm output` | `rms-epsilon` | Logits |
| `rmsnorm` | either | `norm` | `rms-epsilon` | Normalised input |

Optional hparams default as follows: `n-kv-heads` is `n-heads`, `head-dim` is `dim / n-heads`, `rope-dims` is `head-dim`, and `rope-theta` is 10000.

```lisp
(local:define-architecture llama ()
  (:hparams (n-layers) (dim) (n-heads) (n-kv-heads) (rms-epsilon) (rope-theta :default 10000.0))
  (:blocks (attn local:attention) (mlp local:swiglu-mlp))
  (:model-blocks (token local:embedding) (head local:lm-head))
  (:format :gguf
    (:arch "llama")
    (:rope-pairing :adjacent)
    (:tensor attn.q "blk.~d.attn_q.weight")
    (:tensor head.output "output.weight" :fallback token.embd)))
```

## Tensor roles

A block reads the role `<slot>.<suffix>`: slot `attn` reads `attn.q`, `attn.k` and so on. Roles and hparams are matched by symbol name, so blocks work with architectures defined in any package.

`:fallback` names a role to read when the file lacks the tensor, which is how a tied LM head uses the token embedding.

## Layers

`(:blocks ...)` run once per layer in slot order, each adding its returned delta to the residual stream. `(:model-blocks ...)` run once: `embedding` before the layers and `lm-head` after. `layer-forward` specialises on the architecture to change how a layer combines its blocks.

## RoPE

`attention` rotates the first `rope-dims` of each head. The `:rope-pairing` format option picks the layout:

| Pairing | Rotated pairs | Used by |
|---|---|---|
| `:adjacent` (default) | `(2i, 2i+1)` | GGUF `llama`, whose Q/K weights are pre-permuted |
| `:neox` | `(i, i + rope-dims/2)` | Qwen |

Frequency `i` is `rope-theta^(-2i/rope-dims)` divided by element `i` of the optional `rope-freqs` tensor (`rope_freqs.weight` in Llama 3.x files).

## Cache

`block-forward` on `attention` takes `:start` (the position of the first input row) and `:cache`. Without a cache the input attends only to itself. With one, the new keys and values go through `kv-cache-update`. A [session](sessions.md) supplies a `kv-cache`; any object works if it implements the method:

```lisp
(defmethod local:kv-cache-update ((cache my-cache) layer k v start)
  ...                    ; store K and V (positions kv-heads head-dim)
  (values all-k all-v))  ; every position so far
```

## Functions

| Function | Returns |
|---|---|
| `load-model weights` | A `model` with every block bound |
| `model-forward model tokens &key start cache all-logits` | Logits of the last position, or of every position with `all-logits` |
| `block-forward block input &key start cache` | The block's output |
| `layer-forward architecture blocks x &key start cache` | The residual stream after one layer |
| `bind-block block architecture weights` | Reads the block's tensors and hparams |
| `block-tensor block architecture weights suffix` | The tensor for `<slot>.<suffix>` |
| `linear x weight` | `x · weightᵀ`; every projection goes through it. Half-precision weights are multiplied in row tiles[^tiles]; [Q8_0](quant.md) weights by a native kernel |

## Mods

Specialise `block-forward`, including with `:around`, to observe or change a block:

```lisp
(defmethod local:block-forward :around ((block local:attention) x &key &allow-other-keys)
  (let ((delta (call-next-method)))
    (record-activation (local:block-layer block) delta)
    delta))
```

## Limitations

| Missing | Tracker |
|---|---|
| Q8_0 matmul over many activation rows is slower than dequantized tiles through BLAS | [#41](https://github.com/communal-software/cl-inference/issues/41) |

[^tiles]: `linear` converts 256 weight rows at a time to an f32 scratch and multiplies each tile with BLAS, so no more than a tile is ever f32. It converts with trivial-simd's flat `convert!`, which is far faster than `astype` on file-mapped storage.
