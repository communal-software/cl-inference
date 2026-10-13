(in-package #:cl-inference/acceptance)

(in-suite :cl-inference/acceptance)

(setf *cases* (append *cases*
  '((:f16 :layers 16 :model :f16 :text "The capital of France is")
    (:f16-chat :layers 16 :model :f16 :date "26 Jul 2024"
     :messages ((:role :system :content "Answer in one word.")
                (:role :user :content "What is the capital of France?")))
    (:q8 :layers 16 :model :q8_0 :text "The capital of France is")
    (:q8-chat :layers 16 :model :q8_0 :date "26 Jul 2024"
     :messages ((:role :system :content "Answer in one word.")
                (:role :user :content "What is the capital of France?"))))))

(setf *models* (append *models*
  '((:f16 "Llama-3.2-1B-Instruct-f16.gguf"
     "1f33ad43d2b85b908ff06fe7002b69806a57359b9b2617ca27d7bdea428ae146" :dtype :f16)
    (:q8_0 "Llama-3.2-1B-Instruct-Q8_0.gguf"
     "432f310a77f4650a88d0fd59ecdd7cebed8d684bafea53cbff0473542964f0c3" :dtype :q8_0))))

;;; Metadata

(test llama-3-2-1b-f16-metadata
  (if (not (verified-model-p :f16))
      (skip "Llama-3.2-1B-Instruct-f16.gguf is not in reference/models")
      (local:with-weights (weights (model-path :f16))
        (let ((architecture (local:load-architecture weights)))
          (is (eq 'local:llama (type-of architecture)))
          (is (= 500000 (round (local:hparam architecture 'rope-theta))))
          (is (= 16 (local:hparam architecture 'n-layers)))
          (is (member "rope_freqs.weight" (local:weights-tensor-names weights) :test #'string=))
          (is (not (member "output.weight" (local:weights-tensor-names weights) :test #'string=)))
          (is (equal "llama-bpe" (local:weights-metadata weights "tokenizer.ggml.pre")))
          (is (eq 'local:llama-bpe (type-of (local:load-tokenizer weights))))
          (is (eq 'local:llama3-chat (type-of (local:load-chat-template weights))))
          (is (member 128009 (local:chat-template-stop-ids (local:load-chat-template weights)
                                                           (local:load-tokenizer weights))))))))

(define-case-tests :f16)
(define-case-tests :f16-chat)
(define-case-tests :q8)
(define-case-tests :q8-chat)
