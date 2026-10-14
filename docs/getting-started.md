# Getting started

Load `cl-inference/quant`, open a supported GGUF model, and pass a local backend to `complete`. Keep its weights open throughout the request.

## Install

Use SBCL for local inference. Install Quicklisp, Git and CMake, then put these checkouts in ASDF's `~/common-lisp/` search directory:

```sh
mkdir -p ~/common-lisp
git clone https://github.com/communal-software/trivial-simd ~/common-lisp/trivial-simd
git clone https://github.com/communal-software/cl-tensor ~/common-lisp/cl-tensor
git clone https://github.com/communal-software/cl-inference ~/common-lisp/cl-inference
cmake -S ~/common-lisp/trivial-simd -B ~/common-lisp/trivial-simd/build -DCMAKE_BUILD_TYPE=Release
cmake --build ~/common-lisp/trivial-simd/build -j8
```

The native library accelerates inference; quantized kernels have a slower Lisp fallback.[^kernels] In a fresh SBCL image with Quicklisp loaded:

```lisp
(ql:quickload "cl-inference/quant")
```

For model downloads and verification, follow [Model acceptance](acceptance.md). Llama 3.2 1B Instruct and Qwen3 0.6B use the shipped architectures, tokenizers and chat templates.

## Make one request

Replace the pathname with your GGUF file. `cl-inference/quant` includes the GGUF reader and Q8_0, Q4_0, Q4_K and Q6_K support.

```lisp
(cl-inference/local:with-weights (weights "Qwen3-0.6B-Q4_K_M.gguf")
  (let ((backend (cl-inference/local:make-local-backend weights :n-ctx 2048)))
    (unwind-protect
         (let ((result (cl-inference:complete
                        backend :messages '((:role :user :content "Hello!"))
                        :temperature 0 :max-tokens 64 :timeout 120000)))
           (if (eq :ok (first result))
               (dolist (block (getf (second result) :content))
                 (when (eq :text (getf block :type))
                   (write-string (getf block :text))))
               (format t "Request failed: ~S~%" (second result))))
      (cl-inference/local:close-local-backend backend))))
```

`complete` returns `(:ok reply)` or `(:error reason)`. Text is in the reply's `:content` blocks; usage and the finish reason are in `:meta`. `complete*` returns the reply directly and signals a condition on failure. See [Client](client.md) for streaming and tools.

## Choose a system

| Task | Load |
|---|---|
| HTTP requests only | `cl-inference` |
| GGUF inspection or floating-point inference | `cl-inference/gguf` |
| GGUF quantized inference | `cl-inference/quant` |
| HF Safetensors models | `cl-inference/safetensors` |
| Apple Silicon MLX execution | `cl-inference/mlx` |
| Runnable tutorials | `cl-inference/examples` |

Continue with [Examples](examples.md), [Local backend](local-backend.md), or [Architectures](architectures.md) for an application-defined model.

## Limitations

Local inference is verified on SBCL. Other restrictions are listed in [Support](support.md#limitations).

[^kernels]: `cl-inference` contains Lisp runtime code; native CPU kernels live in `trivial-simd`. Its library loads when present at startup, so restart Lisp after building it.
