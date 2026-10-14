# GGUF

`cl-inference/gguf` reads single-file little-endian GGUF v3 through the [weights protocol](weights.md). Loading the system registers the `:gguf` format; files are detected by their `GGUF` magic.

```lisp
(asdf:load-system :cl-inference/gguf)   ; loads cl-inference/local
(local:with-weights (w "Llama-3.2-1B-Instruct-Q8_0.gguf")
  (local:weights-metadata w "general.architecture"))
;; => "llama", T
```

`weights-architecture-name` returns `general.architecture`, which [`load-architecture`](architectures.md) matches against each architecture's `:gguf` `:arch`.

## Mapping

The file is mapped once, copy-on-write, and every tensor view points into the mapping.[^mmap] Writing through a view changes memory only, never the file. Views are invalid after `close-weights`.

| Platform | Status |
|---|---|
| macOS, Linux | Supported |
| BSD | Untested; see [limitations](#limitations) |
| Windows | Signals an error at load; see [limitations](#limitations) |

## Tensor types

| ggml type | `weights-tensor` | `weights-tensor-bytes` |
|---|---|---|
| F32, F64 | `:f32`, `:f64` view | yes |
| F16, BF16 | `:f16`, `:bf16` view | yes |
| I8, I16, I32, I64 | `:s8`, `:s16`, `:s32`, `:s64` view | yes |
| Q8_0, Q4_0, Q4_K, Q6_K | Corresponding packed dtype view, with [`cl-inference/quant`](quant.md) loaded | yes |
| Every other quantized type | `unsupported-weights-type` until a method exists | yes |

All listed ggml types are bounds- and overlap-checked, including formats without typed inference support.[^types] Types that ggml has removed, and unknown ids, signal `gguf-error` at open.

Shapes are row-major: GGUF stores the innermost dimension first, so a tensor with dimensions `(3 2)` has shape `(2 3)`.

Another system adds a typed view by specializing `gguf-type-storage`, which returns storage and a dtype:

```lisp
(defmethod gguf:gguf-type-storage ((type (eql :q5_0)) pointer element-count)
  (values (make-q5-storage pointer element-count) :q5_0))
```

## Metadata values

| GGUF | Lisp |
|---|---|
| integers, floats | numbers |
| bool | `t` / `nil` |
| string | string |
| numeric array | specialized simple-array, e.g. `(simple-array single-float (*))` |
| string or nested array | simple-vector |

## Validation

A file with metadata and no tensors opens, so a vocabulary-only file such as llama.cpp's `ggml-vocab-*.gguf` is readable.

Opening signals `gguf-error` for:

- a file shorter than the 24-byte header, wrong magic, or any version but 3;
- counts, string lengths or array lengths larger than the remaining file;
- invalid UTF-8, invalid booleans, unknown value types, nesting deeper than 32;
- duplicate metadata keys or tensor names;
- tensor rank outside 1–4, a zero dimension, a row not a whole number of blocks;
- `general.alignment` that is not a power of two up to 4096, unaligned or overlapping tensors, or tensors running past the end of the file, including a tensor-bearing file with no data section.

The file is unmapped again when any check fails.

## Limitations

| Missing | Tracker |
|---|---|
| Windows and BSD mapping | [#20](https://github.com/communal-software/cl-inference/issues/20), [#21](https://github.com/communal-software/cl-inference/issues/21) |
| Split (multi-file) GGUF | not planned |

[^mmap]: `PROT_READ|PROT_WRITE` with `MAP_PRIVATE`, because cl-tensor treats foreign views as writable. The header is parsed straight from the mapping with bounds-checked reads. The file is not read through a stream.
[^types]: Block sizes and byte counts come from ggml's `ggml-common.h`. Metadata `float` values are read as native floats.
