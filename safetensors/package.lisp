(defpackage #:cl-inference/safetensors
  (:use #:cl)
  (:local-nicknames (#:local #:cl-inference/local) (#:ct #:cl-tensor)
                    (#:mapping #:cl-inference/mapping) (#:json #:com.inuoe.jzon))
  (:export #:safetensors-error #:safetensors-weights #:open-safetensors))
