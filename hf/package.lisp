(defpackage #:cl-inference/hf
  (:use #:cl)
  (:local-nicknames (#:local #:cl-inference/local) (#:json #:com.inuoe.jzon))
  (:export #:hf-weights #:read-json-file))
