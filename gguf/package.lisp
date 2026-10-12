(defpackage #:cl-inference/gguf
  (:use #:cl)
  (:local-nicknames (#:local #:cl-inference/local) (#:ct #:cl-tensor))
  (:export #:gguf-error #:gguf-type-storage #:gguf-weights #:gguf-tensor #:gguf-tensor-offset
           #:open-gguf))
