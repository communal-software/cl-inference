(in-package #:cl-inference/mlx/tests)
(def-suite :cl-inference/gemma-hf-mlx/acceptance :in :cl-inference/mlx/acceptance)
(in-suite :cl-inference/gemma-hf-mlx/acceptance)

(test gemma-hf-mlx-real-model
  (let ((data (cl-inference/acceptance::gemma-hf-oracle))
        (before (mx:live-array-count)))
    (if data
        (dolist (name '("plain" "chat" "system" "multi" "window"))
          (format t "~&Gemma HF MLX: ~A~%" name)
          (cl-inference/acceptance::compare-gemma-hf
           name data #'mlx:load-mlx-model #'gemma-traced-prefill
           #'mlx:make-mlx-backend #'mlx:close-mlx-backend))
        (skip "Pinned Gemma HF references are absent"))
    (is (= before (mx:live-array-count)))))
