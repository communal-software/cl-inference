(defpackage #:cl-inference/mlx-gguf/tests
  (:use #:cl #:fiveam)
  (:local-nicknames (#:local #:cl-inference/local) (#:mlx #:cl-inference/mlx)
                    (#:mx #:cl-mlx) (#:ct #:cl-tensor) (#:ci #:cl-inference)
                    (#:f #:cl-inference/local/tests) (#:bridge #:cl-mlx/gpu))
  (:export #:run-tests #:run-acceptance))
(in-package #:cl-inference/mlx-gguf/tests)
(def-suite :cl-inference/mlx-gguf)
(defun run-tests () (run! :cl-inference/mlx-gguf))
