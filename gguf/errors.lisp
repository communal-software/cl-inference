(in-package #:cl-inference/gguf)

(define-condition gguf-error (ci:weights-error) ())

(defun gguf-fail (control &rest args)
  (apply #'ci:weights-fail 'gguf-error control args))
