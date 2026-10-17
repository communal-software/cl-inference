(defpackage #:cl-inference/safetensors
  (:use #:cl)
  (:local-nicknames (#:local #:cl-inference/local) (#:ct #:cl-tensor)
                    (#:mapping #:cl-inference/mapping) (#:json #:com.inuoe.jzon))
  (:import-from #:cl-inference/hf #:hf-weights #:safetensors-error #:fail
                #:+maximum-json-bytes+ #:parse-json-object #:read-json-file
                #:*llama-pattern* #:*qwen-pattern*)
  (:export #:safetensors-error #:safetensors-weights #:open-safetensors))
