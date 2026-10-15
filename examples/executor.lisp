(in-package #:cl-inference/examples)

(defun pool-executor (pool run-pool)
  "Adapt a caller-owned synchronous pool whose jobs receive (worker-id worker-count)."
  (lambda (count callback)
    (funcall run-pool pool
             (lambda (id workers)
               (funcall callback (floor (* count id) workers)
                        (floor (* count (1+ id)) workers))))))
