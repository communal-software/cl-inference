(require :asdf)
(asdf:load-system "cl-inference/onnx")

(defun onnx-chat (directory)
  (cl-inference/onnx:with-onnx-backend (backend directory :graph "onnx/model.onnx")
    (cl-inference:complete backend :messages '((:role :user :content "Hello!"))
                                  :temperature 0 :max-tokens 32 :stream #'print)))
