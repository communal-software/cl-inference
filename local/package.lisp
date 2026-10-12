(defpackage #:cl-inference/local
  (:use #:cl)
  (:local-nicknames (#:ci #:cl-inference) (#:ct #:cl-tensor) (#:mop #:closer-mop))
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
   #:architecture-block-spec #:architecture-model-block-slots
   #:architecture-model-block-spec #:architecture-format-option #:architecture-tensor-name
   #:architecture-tensor
   #:architecture-error #:architecture-error-message #:unknown-architecture
   #:unknown-architecture-name #:llama
   ;; tokenizers
   #:define-tokenizer #:tokenizer #:tokenizer-class #:byte-level-bpe #:load-tokenizer
   #:tokenizer-format #:tokenizer-format-option #:tokenizer-pre-rules #:rule-matcher
   #:tokenizer-selector #:tokenizer-load #:pre-tokenize
   #:encode #:decode #:token-id #:token-bytes #:tokenizer-vocab-size #:tokenizer-special-id
   #:tokenizer-special-p #:tokenizer-add-bos-p #:tokenizer-ignore-merges-p
   #:make-detokenizer #:detokenizer-push #:detokenizer-finish
   #:tokenizer-error #:tokenizer-error-message #:unknown-tokenizer #:unknown-tokenizer-model
   #:unknown-tokenizer-pre #:llama-bpe #:qwen2
   ;; chat templates
   #:define-chat-template #:chat-template #:chat-template-class #:load-chat-template
   #:render-chat #:encode-chat #:chat-template-stop-ids #:chat-template-source
   #:chat-template-error #:chat-template-error-message #:unknown-chat-template
   #:unknown-chat-template-source #:llama3-chat #:chatml #:qwen3-chat
   ;; blocks and models
   #:model-block #:block-slot #:block-layer #:bind-block #:block-forward #:block-tensor
   #:kv-cache-update #:linear #:rmsnorm #:embedding #:attention #:swiglu-mlp #:lm-head
   #:model #:model-architecture #:model-layers #:model-model-blocks #:load-model
   #:layer-forward #:model-forward
   ;; sessions
   #:kv-cache #:make-kv-cache #:make-session #:session #:session-model #:session-tokenizer
   #:session-chat-template #:session-n-ctx #:session-n-batch #:session-position #:session-reset
   #:session-prefill #:generate #:greedy-sampler #:stop-split
   #:session-error #:session-error-message #:context-overflow
   #:sample #:sampler #:make-sampler #:sampler-temperature #:sampler-top-k #:sampler-top-p
   #:sampler-seed #:sampler-error
   ;; client backend
   #:local-backend))
