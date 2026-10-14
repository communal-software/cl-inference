(defpackage #:cl-inference/mlx
  (:use #:cl)
  (:local-nicknames (#:local #:cl-inference/local) (#:mx #:cl-mlx)
                    (#:ct #:cl-tensor) (#:st #:cl-inference/safetensors)
                    (#:ci #:cl-inference) (#:bt #:bordeaux-threads-2))
  (:export #:load-mlx-model #:make-mlx-backend #:mlx-model #:mlx-backend
           #:close-mlx-backend #:with-mlx-backend))
