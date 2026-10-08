(defpackage #:cl-inference/tests
  (:use #:cl #:fiveam)
  (:export #:run-tests))

(in-package #:cl-inference/tests)

(def-suite :cl-inference)

(defun run-tests ()
  (run! :cl-inference))
