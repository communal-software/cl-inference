(defpackage #:cl-inference/tests
  (:use #:cl #:fiveam)
  (:local-nicknames (#:ci #:cl-inference) (#:ct #:cl-tensor)
                    (#:gguf #:cl-inference/gguf))
  (:export #:run-tests))

(in-package #:cl-inference/tests)

(def-suite :cl-inference)

(defun run-tests ()
  (and (run! :cl-inference)
       (uiop:symbol-call :cl-inference/client/tests :run-tests)))
