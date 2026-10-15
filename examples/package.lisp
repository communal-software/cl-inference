(defpackage #:cl-inference/examples
  (:use #:cl)
  (:local-nicknames (#:ci #:cl-inference) (#:local #:cl-inference/local) (#:ct #:cl-tensor))
  (:export #:pool-executor #:complete-file #:chat-file #:inspect-weights
           #:observed-llama #:observing-attention #:forward-observed))
