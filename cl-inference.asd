(asdf:defsystem "cl-inference"
  :description "Extensible LLM inference for Common Lisp, built on cl-tensor"
  :author "George Watson"
  :license "GPL-3.0-only"
  :version "0.0.0"
  :depends-on ("cl-tensor" "closer-mop")
  :serial t
  :components ((:file "package") (:file "weights") (:file "architecture"))
  :in-order-to ((asdf:test-op (asdf:test-op "cl-inference/tests"))))

(asdf:defsystem "cl-inference/gguf"
  :description "GGUF weight-file reader for cl-inference"
  :author "George Watson"
  :license "GPL-3.0-only"
  :depends-on ("cl-inference" "cl-tensor" "trivial-simd" "cffi" "babel")
  :serial t
  :components ((:file "gguf/package") (:file "gguf/errors") (:file "gguf/posix")
               (:file "gguf/reader") (:file "gguf/weights")))

(asdf:defsystem "cl-inference/quant"
  :description "Quantized dtypes for cl-inference"
  :author "George Watson"
  :license "GPL-3.0-only"
  :depends-on ("cl-inference" "trivial-simd")
  :serial t
  :components ((:file "quant/package")))

(asdf:defsystem "cl-inference/client"
  :description "Model client contract and HTTP backends for cl-inference"
  :author "George Watson"
  :license "GPL-3.0-only"
  :depends-on ("alexandria" "bordeaux-threads" "com.inuoe.jzon" "drakma" "flexi-streams"
               "usocket" "puri" "chunga" "cl+ssl" "uiop")
  :serial t
  :components ((:file "client/package") (:file "client/result") (:file "client/cancel")
               (:file "client/schema") (:file "client/contract") (:file "client/transport")
               (:file "client/backend") (:file "client/openai") (:file "client/ollama")
               (:file "client/provider") (:file "client/providers") (:file "client/scripted"))
  :in-order-to ((asdf:test-op (asdf:test-op "cl-inference/client/tests"))))

(asdf:defsystem "cl-inference/client/tests"
  :depends-on ("cl-inference/client" "fiveam" "usocket" "bordeaux-threads" "flexi-streams")
  :serial t
  :components ((:file "tests/client/package") (:file "tests/client/fake-http")
               (:file "tests/client/schema") (:file "tests/client/transport")
               (:file "tests/client/openai") (:file "tests/client/ollama")
               (:file "tests/client/scripted") (:file "tests/client/provider") (:file "tests/client/systems"))
  :perform (asdf:test-op (op component)
             (declare (ignore op component))
             (unless (uiop:symbol-call :cl-inference/client/tests :run-tests)
               (error "cl-inference/client tests failed"))))

(asdf:defsystem "cl-inference/tests"
  :depends-on ("cl-inference" "cl-inference/gguf" "cl-inference/quant" "cl-inference/client/tests"
               "fiveam")
  :serial t
  :components ((:file "tests/package") (:file "tests/systems") (:file "tests/weights")
               (:file "tests/gguf-writer") (:file "tests/gguf") (:file "tests/architecture"))
  :perform (asdf:test-op (op component)
             (declare (ignore op component))
             (unless (uiop:symbol-call :cl-inference/tests :run-tests)
               (error "cl-inference tests failed"))))
