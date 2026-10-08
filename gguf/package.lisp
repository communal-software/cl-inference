(defpackage #:cl-inference/gguf
  (:use #:cl)
  (:local-nicknames (#:ci #:cl-inference) (#:ct #:cl-tensor))
  (:export #:gguf-error #:gguf-type-storage #:gguf-weights #:gguf-tensor #:gguf-tensor-offset
           #:open-gguf))
