(in-package #:cl-inference/gguf)

(define-condition gguf-error (local:weights-error) ())

(defun gguf-fail (control &rest args)
  (apply #'local:weights-fail 'gguf-error control args))
