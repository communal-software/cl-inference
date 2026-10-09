(in-package #:cl-inference/tests)

(in-suite :cl-inference)

(test packages-load
  (dolist (name '("CL-INFERENCE" "CL-INFERENCE/GGUF" "CL-INFERENCE/QUANT" "CL-INFERENCE/CLIENT"))
    (is (find-package name))))
