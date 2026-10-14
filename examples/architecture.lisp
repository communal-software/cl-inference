(in-package #:cl-inference/examples)

(defvar *observations* nil)

(defclass observing-attention (local:attention) ())

(defmethod local:block-forward :around ((block observing-attention) x &key &allow-other-keys)
  (let ((delta (call-next-method)))
    (push (list :layer (local:block-layer block) :shape (coerce (ct:tensor-shape delta) 'list))
          *observations*)
    delta))

(local:define-architecture observed-llama (local:llama)
  (:blocks (attn observing-attention))
  (:format :gguf (:arch "observed-llama"))
  (:format :safetensors (:arch nil)))

(defun forward-observed (path tokens)
  "Run an observed-llama fixture; return detached logits and per-layer shapes."
  (local:with-weights (weights path)
    (let ((model (local:load-model weights)) (*observations* nil))
      (unwind-protect
           (values (local:model-forward model tokens) (nreverse *observations*))
        (local:close-model model)))))
