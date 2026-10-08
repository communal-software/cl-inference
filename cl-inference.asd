(asdf:defsystem "cl-inference"
  :description "Extensible LLM inference for Common Lisp, built on cl-tensor"
  :author "George Watson"
  :license "GPL-3.0-only"
  :version "0.0.0"
  :depends-on ("cl-tensor")
  :serial t
  :components ((:file "package") (:file "weights"))
  :in-order-to ((asdf:test-op (asdf:test-op "cl-inference/tests"))))

(asdf:defsystem "cl-inference/gguf"
  :description "GGUF weight-file reader for cl-inference"
  :author "George Watson"
  :license "GPL-3.0-only"
  :depends-on ("cl-inference" "cffi")
  :serial t
  :components ((:file "gguf/package")))

(asdf:defsystem "cl-inference/quant"
  :description "Quantized dtypes for cl-inference"
  :author "George Watson"
  :license "GPL-3.0-only"
  :depends-on ("cl-inference" "trivial-simd")
  :serial t
  :components ((:file "quant/package")))

(asdf:defsystem "cl-inference/tests"
  :depends-on ("cl-inference" "cl-inference/gguf" "cl-inference/quant" "fiveam")
  :serial t
  :components ((:file "tests/package") (:file "tests/systems") (:file "tests/weights"))
  :perform (asdf:test-op (op component)
             (declare (ignore op component))
             (unless (uiop:symbol-call :cl-inference/tests :run-tests)
               (error "cl-inference tests failed"))))
