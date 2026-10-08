(defpackage #:cl-inference
  (:use #:cl)
  (:local-nicknames (#:ct #:cl-tensor) (#:mop #:closer-mop))
  (:export
   ;; weights protocol
   #:open-weights #:close-weights #:with-weights #:register-weights-format
   #:weights #:weights-format #:weights-path #:weights-closed-p
   #:weights-metadata #:weights-metadata-keys #:weights-tensor-names
   #:weights-tensor-info #:weights-tensor #:weights-tensor-bytes
   #:tensor-info #:make-tensor-info #:tensor-info-name #:tensor-info-shape
   #:tensor-info-format-type #:tensor-info-byte-count
   ;; format implementers
   #:weights-metadata-ref #:weights-metadata-key-list #:weights-tensor-info-list
   #:weights-tensor-info-ref #:weights-make-tensor #:weights-make-bytes #:weights-release
   #:weights-architecture-name
   ;; conditions
   #:weights-error #:weights-error-message #:weights-fail #:unknown-weights-format
   #:weights-closed #:unsupported-weights-type
   ;; architectures
   #:define-architecture #:architecture #:architecture-class #:load-architecture
   #:hparam #:architecture-format #:architecture-hparam-specs #:architecture-block-slots
   #:architecture-block-spec #:architecture-format-option #:architecture-tensor-name
   #:architecture-tensor
   #:architecture-error #:architecture-error-message #:unknown-architecture
   #:unknown-architecture-name))
