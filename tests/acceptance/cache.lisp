(in-package #:cl-inference/acceptance)
(in-suite :cl-inference/acceptance)

(test q8-cache-real-model-acceptance
  (dolist (case '(("cpu-llama" :f16) ("cpu-qwen" :qwen-f16)))
    (destructuring-bind (name kind) case
      (let ((oracle (tests::read-cache-oracle name)))
        (if (and oracle (verified-model-p kind))
            (local:with-weights (weights (model-path kind))
              (tests::check-q8-cache-oracle name (local:load-model weights) oracle))
            (skip "Missing ~A inputs; cache acceptance is not established" name))))))
