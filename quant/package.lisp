(defpackage #:cl-inference/quant
  (:use #:cl)
  (:local-nicknames (#:ct #:cl-tensor) (#:gguf #:cl-inference/gguf) (#:simd #:trivial-simd))
  (:export #:q8-0 #:q8-0-storage #:make-q8-0-storage #:q8-0-storage-bytes))
