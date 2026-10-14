(in-package #:cl-inference/examples)

(defun complete-file (path messages &key (max-tokens 64) (timeout 30000))
  "Return one greedy reply, releasing the backend before its weights close."
  (local:with-weights (weights path)
    (let ((backend (local:make-local-backend weights :n-ctx 2048)))
      (unwind-protect
           (ci:complete* backend :messages messages :temperature 0
                                 :max-tokens max-tokens :timeout timeout)
        (local:close-local-backend backend)))))
