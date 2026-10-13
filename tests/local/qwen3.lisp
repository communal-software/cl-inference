(in-package #:cl-inference/local/tests)

(in-suite :cl-inference/local)

(test qwen3-inherits-llama-and-overrides-attention
  (is (equal '(local:attention :qk-norm t)
             (local:architecture-block-spec 'local:qwen3 'attn)))
  (is (equal '(local:attention) (local:architecture-block-spec 'local:llama 'attn)))
  (is (equal (local:architecture-block-spec 'local:llama 'mlp)
             (local:architecture-block-spec 'local:qwen3 'mlp)))
  (is (equal (local:architecture-model-block-slots 'local:llama)
             (local:architecture-model-block-slots 'local:qwen3)))
  (let ((architecture (make-instance 'local:qwen3 :format :gguf)))
    (is (equal "blk.3.attn_q_norm.weight"
               (local:architecture-tensor-name architecture 'attn.q-norm 3)))
    (is (equal "blk.3.attn_k_norm.weight"
               (local:architecture-tensor-name architecture 'attn.k-norm 3)))))

(test qwen3-matches-reference-with-explicit-head-dimension
  (dolist (dtype '(:f32 :f16 :bf16))
    (dolist (tied '(nil t))
      (check-model "qwen3" :neox
                   (spec* :qk-norm t :heads 2 :kv 1 :head-dim 8 :rope-dims 8
                          :dtype dtype :tied tied)))))

(test qwen3-q8-matches-reference
  (check-model "qwen3" :neox
               (spec* :qk-norm t :dtype :q8_0 :dim 32 :heads 2 :kv 1
                      :head-dim 32 :rope-dims 32 :ffn 32 :vocab 32 :tied t)))

(test qwen3-cached-decoding-matches-reference
  (let* ((spec (spec* :qk-norm t :head-dim 8 :rope-dims 8))
         (table (weight-table spec)) (tokens '(3 0 5 2 6)))
    (with-spec-model (model "qwen3" spec table)
      (let ((session (local:make-session model :n-batch 2)))
        (local:session-prefill session (subseq tokens 0 3))
        (is (rows-close-p (tensor-rows (local:session-prefill session (subseq tokens 3)))
                          (last (ref-logits table spec :neox tokens))))))))

(test qwen3-requires-both-qk-norm-weights
  (let* ((spec (spec* :qk-norm t)) (table (weight-table spec)))
    (dolist (missing '("blk.0.attn_q_norm.weight" "blk.0.attn_k_norm.weight"))
      (remhash missing table)
      (with-gguf-file (path (build-gguf
                             :metadata '(("general.architecture" 8 "qwen3")
                                         ("qwen3.block_count" 4 1)
                                         ("qwen3.embedding_length" 4 8)
                                         ("qwen3.attention.head_count" 4 2)
                                         ("qwen3.attention.head_count_kv" 4 1)
                                         ("qwen3.attention.layer_norm_rms_epsilon" 6 1f-5))
                             :tensors (loop for name being the hash-keys of table
                                            using (hash-value entry)
                                            collect (list name (reverse (car entry)) 0
                                                          (f32-octets (cdr entry))))))
        (local:with-weights (weights path)
          (signals local:weights-error (local:load-model weights))))
      (setf table (weight-table spec)))))

(test qwen3-template-fallback-and-stop
  (let* ((spec (spec* :qk-norm t)) (table (weight-table spec)))
    (with-gguf-file (path (spec-gguf "qwen3" spec table))
      (local:with-weights (weights path)
        (let* ((template (local:load-chat-template weights))
               (tokenizer (with-gguf-file
                              (vocab (tokenizer-file :pre "qwen2" :add-bos nil
                                                     :tokens (append (fixture-tokens) '("<|im_end|>"))
                                                     :types (append (fixture-types) '(3))
                                                     :specials '(("eos_token_id" 267))))
                            (local:with-weights (w vocab) (local:load-tokenizer w))))
               (session (local:make-session (local:load-model weights)
                                            :tokenizer tokenizer :chat-template template))
               (stop (local:token-id tokenizer "<|im_end|>")))
          (is (eq 'local:qwen3-chat (type-of template)))
          (is (integerp stop))
          (multiple-value-bind (text finish ids)
              (local:generate session '(1) :max-tokens 2 :sampler (script-sampler (list stop)))
            (is (equal "" text))
            (is (eq :stop finish))
            (is (null ids))))))))
