(defpackage #:cl-inference/local/tests
  (:use #:cl #:fiveam)
  (:local-nicknames (#:ci #:cl-inference) (#:local #:cl-inference/local) (#:ct #:cl-tensor)
                    (#:bt #:bordeaux-threads-2) (#:gguf #:cl-inference/gguf) (#:st #:cl-inference/safetensors))
  (:export #:run-tests))

(in-package #:cl-inference/local/tests)

(def-suite :cl-inference/local)

(defun run-tests ()
  (run! :cl-inference/local))
