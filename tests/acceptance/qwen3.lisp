(in-package #:cl-inference/acceptance)

(in-suite :cl-inference/acceptance)

(setf *models*
      (append *models*
              '((:qwen-f16 "Qwen3-0.6B-f16.gguf"
                 "25fb880334987a71db728fb43af8568a4682208059f754a2735781f5d0970143"
                 :dtype :f16 :metadata-test qwen-f16-metadata)
                (:qwen-q8 "Qwen3-0.6B-Q8_0.gguf"
                 "9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031"
                 :dtype :q8_0 :metadata-test qwen-q8-metadata))))

(setf *cases*
      (append *cases*
              '((:qwen-f16 :model :qwen-f16 :layers 28 :text "The capital of France is")
                (:qwen-q8 :model :qwen-q8 :layers 28
                 :tolerances (0.999d0 0.04d0 1) :unordered-top 3 :text "The capital of France is")
                (:qwen-f16-thinking :model :qwen-f16 :layers 28 :thinking t
                 :messages ((:role :system :content "Answer in one word.")
                            (:role :user :content "What is the capital of France?")))
                (:qwen-q8-thinking :model :qwen-q8 :layers 28 :thinking t
                 :messages ((:role :system :content "Answer in one word.")
                            (:role :user :content "What is the capital of France?")))
                (:qwen-f16-chat :model :qwen-f16 :layers 28 :thinking nil
                 :messages ((:role :system :content "Answer in one word.")
                            (:role :user :content "What is the capital of France?")))
                (:qwen-q8-chat :model :qwen-q8 :layers 28 :thinking nil
                 :tolerances (0.998d0 0.06d0 3)
                 :messages ((:role :system :content "Answer in one word.")
                            (:role :user :content "What is the capital of France?"))))))

(defun check-qwen-metadata (kind)
  (if (not (verified-model-p kind))
      (skip "~A is missing or has the wrong sha256" (model-path kind))
      (local:with-weights (weights (model-path kind))
        (let* ((architecture (local:load-architecture weights))
               (tokenizer (local:load-tokenizer weights))
               (template (local:load-chat-template weights)))
          (is (eq 'local:qwen3 (type-of architecture)))
          (is (typep architecture 'local:llama))
          (loop for (name value) in '((n-layers 28) (dim 1024) (n-heads 16)
                                      (n-kv-heads 8) (head-dim 128)
                                      (rope-theta 1000000))
                do (is (= value (local:hparam architecture name))))
          (is (= 128 (or (local:hparam architecture 'rope-dims)
                          (local:hparam architecture 'head-dim))))
          (is (eq :neox (local:architecture-format-option architecture :gguf :rope-pairing)))
          (is (eq 'local:qwen2 (type-of tokenizer)))
          (is (not (local:tokenizer-add-bos-p tokenizer)))
          (is (eq 'local:qwen3-chat (type-of template)))
          (is (member (local:token-id tokenizer "<|im_end|>")
                      (local:chat-template-stop-ids template tokenizer)))
          (dotimes (layer 28)
            (dolist (role '(attn.q-norm attn.k-norm))
              (is (equalp #(128) (ct:tensor-shape
                                 (local:architecture-tensor architecture weights role layer))))))))))

(test qwen-f16-metadata (check-qwen-metadata :qwen-f16))
(test qwen-q8-metadata (check-qwen-metadata :qwen-q8))

(define-case-tests :qwen-f16)
(define-case-tests :qwen-q8)
(define-case-tests :qwen-f16-thinking)
(define-case-tests :qwen-q8-thinking)
(define-case-tests :qwen-f16-chat)
(define-case-tests :qwen-q8-chat)
