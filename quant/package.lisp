(defpackage #:cl-inference/quant
  (:use #:cl)
  (:local-nicknames (#:ct #:cl-tensor) (#:gguf #:cl-inference/gguf) (#:simd #:trivial-simd))
  (:export #:q8-0 #:q8-0-storage #:make-q8-0-storage #:q8-0-storage-bytes
           #:q4-0 #:q4-0-storage #:make-q4-0-storage #:q4-0-storage-bytes
           #:q4-k #:q4-k-storage #:make-q4-k-storage #:q4-k-storage-bytes
           #:q6-k #:q6-k-storage #:make-q6-k-storage #:q6-k-storage-bytes))
