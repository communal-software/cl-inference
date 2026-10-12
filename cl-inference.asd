(asdf:defsystem "cl-inference"
  :description "Model client for Common Lisp: one contract for hosted and local models"
  :author "George Watson"
  :license "GPL-3.0-only"
  :version "0.0.0"
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
  :depends-on ("cl-inference" "cl-tensor" "closer-mop")
  :serial t
  :components ((:file "local/package") (:file "local/weights") (:file "local/architecture")
               (:file "local/backend"))
  :in-order-to ((asdf:test-op (asdf:test-op "cl-inference/local/tests"))))

(asdf:defsystem "cl-inference/gguf"
  :description "GGUF weight-file reader for cl-inference/local"
  :author "George Watson"
  :license "GPL-3.0-only"
  :depends-on ("cl-inference/local" "cl-tensor" "trivial-simd" "cffi" "babel")
  :serial t
  :components ((:file "gguf/package") (:file "gguf/errors") (:file "gguf/posix")
               (:file "gguf/reader") (:file "gguf/weights")))

(asdf:defsystem "cl-inference/quant"
  :description "Quantized dtypes for cl-inference/local"
  :author "George Watson"
  :license "GPL-3.0-only"
  :depends-on ("cl-inference/local" "trivial-simd")
  :serial t
  :components ((:file "quant/package")))

(asdf:defsystem "cl-inference/local/tests"
  :depends-on ("cl-inference/local" "cl-inference/gguf" "cl-inference/quant" "fiveam")
  :serial t
  :components ((:file "tests/local/package") (:file "tests/local/systems")
               (:file "tests/local/weights") (:file "tests/local/gguf-writer")
               (:file "tests/local/gguf") (:file "tests/local/architecture"))
  :perform (asdf:test-op (op component)
             (declare (ignore op component))
             (unless (uiop:symbol-call :cl-inference/local/tests :run-tests)
               (error "cl-inference/local tests failed"))))
