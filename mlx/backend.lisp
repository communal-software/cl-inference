(in-package #:cl-inference/mlx)

(defclass mlx-backend (local:local-backend) ())

(defun make-mlx-backend (weights &key (name :mlx) n-ctx (n-batch 512) (session-policy :serialized))
  "A client backend borrowing open Safetensors WEIGHTS."
  (unless (member session-policy '(:serialized :per-call))
    (local::session-fail "session-policy must be :serialized or :per-call"))
  (let ((model (load-mlx-model weights)) (success nil))
    (unwind-protect
         (let* ((backend (make-instance 'mlx-backend :name name :weights weights :model model
                                       :tokenizer (local:load-tokenizer weights)
                                       :template (local:load-chat-template weights)
                                       :n-ctx (or n-ctx (local::default-n-ctx model)) :n-batch n-batch
                                       :session-policy session-policy))
                (session (local::backend-new-session backend)))
           (when (> (local:tokenizer-vocab-size (local::backend-tokenizer backend))
                    (first (weight-shape (local::embedding-table (local::model-block-of-type model 'local:embedding)))))
             (local:close-session session)
             (model-fail "Tokenizer vocabulary exceeds embedding rows"))
           (if (eq session-policy :serialized)
               (setf (local::backend-session backend) session)
               (local:close-session session))
           (setf success t)
           backend)
      (unless success (local:close-model model)))))

(defun close-mlx-backend (backend) (local:close-local-backend backend))

(defmacro with-mlx-backend ((backend weights &rest options) &body body)
  `(let ((,backend (make-mlx-backend ,weights ,@options)))
     (unwind-protect (progn ,@body) (close-mlx-backend ,backend))))

(defmethod ci:describe-backend ((backend mlx-backend))
  (let ((description (call-next-method)))
    (setf (getf description :summary) "In-process MLX model on Apple Silicon")
    description))

(ci:register-backend (make-instance 'mlx-backend :name :mlx))
