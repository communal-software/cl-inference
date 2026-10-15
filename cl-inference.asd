(asdf:defsystem "cl-inference"
  :description "Model client for Common Lisp: one contract for hosted and local models"
  :author "George Watson"
  :license "GPL-3.0-only"
  :version "0.0.1"
  :depends-on ("alexandria" "bordeaux-threads" "com.inuoe.jzon" "drakma" "flexi-streams"
               "usocket" "puri" "chunga" "cl+ssl" "uiop")
  :serial t
  :components ((:file "package") (:file "result") (:file "cancel")
               (:file "schema") (:file "contract") (:file "transport")
               (:file "backend") (:file "openai") (:file "ollama")
               (:file "provider") (:file "providers") (:file "scripted"))
  :in-order-to ((asdf:test-op (asdf:test-op "cl-inference/tests"))))

(asdf:defsystem "cl-inference/tests"
  :depends-on ("cl-inference" "fiveam" "usocket" "bordeaux-threads" "flexi-streams")
  :serial t
  :components ((:file "tests/package") (:file "tests/fake-http")
               (:file "tests/schema") (:file "tests/transport")
               (:file "tests/openai") (:file "tests/ollama")
               (:file "tests/scripted") (:file "tests/provider") (:file "tests/systems"))
  :perform (asdf:test-op (op component)
             (declare (ignore op component))
             (unless (uiop:symbol-call :cl-inference/tests :run-tests)
               (error "cl-inference tests failed"))))

(asdf:defsystem "cl-inference/local"
  :description "In-process inference on cl-tensor, a local provider for cl-inference"
  :author "George Watson"
  :license "GPL-3.0-only"
  :depends-on ("cl-inference" "cl-tensor" "trivial-simd" "closer-mop" "cl-unicode" "babel" "float-features")
  :serial t
  :components ((:file "local/package") (:file "local/weights") (:file "local/sections")
               (:file "local/architecture") (:file "local/tokenizer")
               (:file "local/pre-tokenizer") (:file "local/bpe") (:file "local/tokenizers")
               (:file "local/chat-template") (:file "local/tojson")
               (:file "local/chat-templates") (:file "local/chat-families")
               (:file "local/blocks") (:file "local/architectures") (:file "local/kv-cache")
               (:file "local/session") (:file "local/sampler")
               (:file "local/reply-parser") (:file "local/backend"))
  :in-order-to ((asdf:test-op (asdf:test-op "cl-inference/local/tests"))))

(asdf:defsystem "cl-inference/mapping"
  :description "Internal POSIX copy-on-write file mappings"
  :license "GPL-3.0-only"
  :depends-on ("cffi")
  :serial t
  :components ((:file "mapping/package") (:file "mapping/posix")))

(asdf:defsystem "cl-inference/safetensors"
  :description "Safetensors and Hugging Face configuration reader"
  :license "GPL-3.0-only"
  :depends-on ("cl-inference/local" "cl-inference/mapping" "trivial-simd" "babel")
  :serial t
  :components ((:file "safetensors/package") (:file "safetensors/reader")
               (:file "safetensors/weights") (:file "safetensors/model")
               (:file "safetensors/tokenizer")))

(asdf:defsystem "cl-inference/gguf"
  :description "GGUF weight-file reader for cl-inference/local"
  :author "George Watson"
  :license "GPL-3.0-only"
  :depends-on ("cl-inference/local" "cl-inference/mapping" "cl-tensor" "trivial-simd" "babel")
  :serial t
  :components ((:file "gguf/package") (:file "gguf/errors") (:file "gguf/posix")
               (:file "gguf/reader") (:file "gguf/weights")))

(asdf:defsystem "cl-inference/quant"
  :description "Quantized dtypes for cl-inference/local"
  :author "George Watson"
  :license "GPL-3.0-only"
  :depends-on ("cl-inference/local" "cl-inference/gguf" "cl-tensor" "trivial-simd")
  :serial t
  :components ((:file "quant/package") (:file "quant/q8-0") (:file "quant/packed")))

(asdf:defsystem "cl-inference/local/tests"
  :depends-on ("cl-inference/local" "cl-inference/gguf" "cl-inference/safetensors" "cl-inference/quant" "fiveam")
  :serial t
  :components ((:file "tests/local/package") (:file "tests/local/systems")
               (:file "tests/local/weights") (:file "tests/local/gguf-writer")
               (:file "tests/local/gguf") (:file "tests/local/fixtures")
               (:file "tests/local/architecture")
               (:file "tests/local/blocks") (:file "tests/local/tokenizer")
               (:file "tests/local/vocab-fixtures")
               (:file "tests/local/chat-template-expected") (:file "tests/local/chat-template")
               (:file "tests/local/session") (:file "tests/local/cache") (:file "tests/local/sampler")
               (:file "tests/local/quant") (:file "tests/local/quant-golden")
               (:file "tests/local/packed-fixtures") (:file "tests/local/packed")
               (:file "tests/local/qwen3")
               (:file "tests/local/safetensors") (:file "tests/local/hf-tokenizer")
               (:file "tests/local/backend") (:file "tests/local/execution")
               (:file "tests/local/chat-families-expected") (:file "tests/local/chat-families"))
  :perform (asdf:test-op (op component)
             (declare (ignore op component))
             (unless (uiop:symbol-call :cl-inference/local/tests :run-tests)
               (error "cl-inference/local tests failed"))))

(asdf:defsystem "cl-inference/local/acceptance"
  :description "Llama and Qwen3 against llama.cpp; needs models and oracle in reference/"
  :depends-on ("cl-inference/local/tests" "uiop" "fiveam")
  :serial t
  :components ((:file "tests/acceptance/package") (:file "tests/acceptance/helpers")
               (:file "tests/acceptance/llama") (:file "tests/acceptance/qwen3")
               (:file "tests/acceptance/quant") (:file "tests/acceptance/cache"))
  :perform (asdf:test-op (op component)
             (declare (ignore op component))
             (unless (uiop:symbol-call :cl-inference/acceptance :run-tests)
               (error "cl-inference acceptance tests failed"))))

(asdf:defsystem "cl-inference/examples"
  :description "Runnable local inference and architecture extension examples"
  :license "GPL-3.0-only"
  :depends-on ("cl-inference/quant")
  :serial t
  :components ((:file "examples/package") (:file "examples/complete")
               (:file "examples/weights") (:file "examples/architecture") (:file "examples/executor")
               (:file "examples/chat"))
  :in-order-to ((asdf:test-op (asdf:test-op "cl-inference/examples/tests"))))

(asdf:defsystem "cl-inference/examples/tests"
  :depends-on ("cl-inference/examples" "cl-inference/local/tests")
  :components ((:file "tests/examples"))
  :perform (asdf:test-op (op component)
             (declare (ignore op component))
             (unless (uiop:symbol-call :cl-inference/examples/tests :run-tests)
               (error "cl-inference examples failed"))))

(asdf:defsystem "cl-inference/mlx"
  :description "Apple Silicon inference on lazy MLX arrays"
  :license "GPL-3.0-only"
  :depends-on ("cl-inference/safetensors" "cl-mlx")
  :serial t
  :components ((:file "mlx/package") (:file "mlx/model") (:file "mlx/backend"))
  :in-order-to ((asdf:test-op (asdf:test-op "cl-inference/mlx/tests"))))

(asdf:defsystem "cl-inference/mlx/tests"
  :depends-on ("cl-inference/mlx" "cl-inference/local/tests")
  :serial t
  :components ((:file "tests/mlx/package") (:file "tests/mlx/model") (:file "tests/mlx/backend")
               (:file "tests/mlx/acceptance-support") (:file "tests/mlx/acceptance-harness"))
  :perform (asdf:test-op (op component)
             (declare (ignore op component))
             (unless (uiop:symbol-call :cl-inference/mlx/tests :run-tests)
               (error "cl-inference/mlx tests failed"))))

(asdf:defsystem "cl-inference/mlx/acceptance"
  :description "Qwen3 float and 4-bit against a pinned Python MLX reference"
  :depends-on ("cl-inference/mlx/tests")
  :serial t
  :components ((:file "tests/mlx/acceptance") (:file "tests/mlx/cache-acceptance"))
  :perform (asdf:test-op (op component)
             (declare (ignore op component))
             (unless (uiop:symbol-call :cl-inference/mlx/tests :run-acceptance)
               (error "cl-inference/mlx acceptance tests failed"))))
