(defpackage #:cl-inference/acceptance
  (:use #:cl #:fiveam)
  (:local-nicknames (#:ci #:cl-inference) (#:local #:cl-inference/local) (#:ct #:cl-tensor)
                    (#:gguf #:cl-inference/gguf) (#:tests #:cl-inference/local/tests))
  (:export #:run-tests))

(in-package #:cl-inference/acceptance)

(defvar *hashes* (make-hash-table))
(defvar *runs* (make-hash-table))

(def-suite :cl-inference/acceptance)

(defun run-tests ()
  (clrhash *hashes*)
  (clrhash *runs*)
  (run! :cl-inference/acceptance))
