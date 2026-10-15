(in-package #:cl-inference/mlx/tests)
(in-suite :cl-inference/mlx/acceptance)

(test mlx-q8-cache-real-model-acceptance
  (dolist (name '("llama-float" "llama-4bit" "qwen-float" "qwen-4bit"))
    (let ((oracle (f::read-cache-oracle name)) (before (mx:live-array-count)))
      (if oracle
          (local:with-weights (weights (merge-pathnames (format nil "~A/" name) (mlx-reference-root)))
            (let ((model (mlx:load-mlx-model weights)))
              (unwind-protect (f::check-q8-cache-oracle name model oracle)
                (local:close-model model))))
          (skip "Missing ~A inputs; cache acceptance is not established" name))
      (is (= before (mx:live-array-count))))))
