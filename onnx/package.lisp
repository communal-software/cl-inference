(defpackage #:cl-inference/onnx
  (:use #:cl)
  (:local-nicknames (#:local #:cl-inference/local) (#:ci #:cl-inference)
                    (#:ox #:cl-onnx) (#:hf #:cl-inference/hf)
                    (#:ct #:cl-tensor) (#:bt #:bordeaux-threads-2))
  (:export #:onnx-error #:onnx-backend #:make-onnx-backend #:close-onnx-backend #:with-onnx-backend))
