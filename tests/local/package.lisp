(defpackage #:cl-inference/local/tests
  (:use #:cl #:fiveam)
  (:local-nicknames (#:ci #:cl-inference) (#:local #:cl-inference/local) (#:ct #:cl-tensor)
                    (#:gguf #:cl-inference/gguf))
  (:export #:run-tests))

(in-package #:cl-inference/local/tests)

(def-suite :cl-inference/local)

(defun run-tests ()
  (run! :cl-inference/local))
