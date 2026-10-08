(defpackage #:cl-inference
  (:use #:cl)
  (:local-nicknames (#:ct #:cl-tensor))
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
   ;; conditions
   #:weights-error #:weights-error-message #:weights-fail #:unknown-weights-format
   #:weights-closed #:unsupported-weights-type))
