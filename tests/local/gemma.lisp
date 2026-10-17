(in-package #:cl-inference/local/tests)
(in-suite :cl-inference/local)

(defun gemma-spec (&rest keys)
  (apply #'spec* :gemma3 t :qk-norm t :layers 6 :head-dim 8 :rope-dims 8
         :window 3 :pattern 3 :local-theta 10000f0 keys))

(test gemma3-prompts-match-pinned-jinja
  (let ((template (make-instance 'local:gemma3-chat)))
    (loop for (messages add expected) in *gemma-template-cases*
          for decoded-messages = (loop for message in messages
                                       collect (list :role (getf message :role)
                                                     :content (unescape (getf message :content))))
          do (if (eq expected :error)
                 (signals local:chat-template-error
                   (local:render-chat template decoded-messages :add-generation-prompt add))
                 (multiple-value-bind (text segments)
                     (local:render-chat template decoded-messages :add-generation-prompt add)
                   (is (string= (unescape expected) text))
                   (is (string= text (local::chat-segments-text segments)))))))
  (let ((source (uiop:read-file-string (asdf:system-relative-pathname
                                       "cl-inference" "tests/local/templates/google-gemma-3-270m-it.jinja"))))
    (is (eq 'local:gemma3-chat (class-name (local::detected-chat-template-class source)))))
  (signals local:unknown-chat-template
    (loaded-chat-template-class :architecture "gemma3" :template "<start_of_turn>")))

(test gemma3-pinned-real-tokenizer-parity
  (let ((path (fixture-path "models/gemma-3-270m-it-Q8_0.gguf")))
    (if (probe-file path)
        (local:with-weights (weights path)
          (let ((tokenizer (local:load-tokenizer weights)))
            (loop for (input add special want hex) in *gemma-tokenizer-cases*
                  for text = (unescape input)
                  for ids = (local:encode tokenizer text :add-bos add :special special)
                  for decoder = (local:make-detokenizer tokenizer :special special)
                  do (is (equal want (coerce ids 'list)) "Gemma ~S (~S/~S)" text add special)
                     (is (string= (babel:octets-to-string (hex-octets hex) :encoding :utf-8 :errorp nil)
                                  (local:decode tokenizer ids :special special)))
                     (is (string= (local:decode tokenizer ids :special special)
                                  (with-output-to-string (out)
                                    (loop for id across ids do (write-string (local:detokenizer-push decoder id) out))
                                    (write-string (local:detokenizer-finish decoder) out)))))))
        (skip "Pinned Gemma vocabulary is absent; real tokenizer parity is not established"))))

(test gemma3-reference-logits-and-cached-decoding
  (dolist (dtype '(:f32 :f16 :bf16 :q8_0))
    (let* ((spec (gemma-spec :dim 32 :head-dim 32 :rope-dims 32 :ffn 32 :vocab 32
                             :dtype dtype :tied t))
           (table (weight-table spec)) (tokens '(3 0 5 2 6 1 4)))
      (with-spec-model (model "gemma3" spec table)
        (let ((expected (ref-logits table spec :neox tokens)))
          (is (rows-close-p (tensor-rows (local:model-forward model tokens :all-logits t)) expected 1d-3))
          (dolist (batch '(1 2 4))
            (dolist (cache-type '(:f32 :q8))
              (let ((session (local:make-session model :n-batch batch :n-ctx 16 :cache-type cache-type)))
                (unwind-protect
                     (progn
                       (local:session-prefill session (subseq tokens 0 4))
                       (is (rows-close-p (tensor-rows (local:session-prefill session (subseq tokens 4)))
                                         (last expected) (if (eq cache-type :q8) 0.1d0 1d-3))))
                  (local:close-session session))))))))))

(test gemma3-window-mask-boundaries-and-absolute-positions
  (let ((mask (local::causal-mask 3 8 5 0 3)))
    (is (equal '((0 0 0 1 1 1 0 0) (0 0 0 0 1 1 1 0) (0 0 0 0 0 1 1 1))
               (tensor-rows mask))))
  (let* ((spec (gemma-spec)) (table (weight-table spec)))
    (with-spec-model (model "gemma3" spec table)
      (loop for blocks in (local:model-layers model) for layer from 0
            for attention = (first blocks)
            do (is (= (if (= 2 (mod layer 3)) 500000f0 10000f0)
                      (local::attn-rope-theta attention)))
               (is (eql (if (= 2 (mod layer 3)) nil 3)
                        (local::gemma3-attention-window attention))))
      (is (typep (local:model-architecture model) 'local:gemma3))
      (is (not (local:architecture-format-option 'local:gemma3 :safetensors :arch))))))

(test gemma3-missing-and-malformed-tensors
  (dolist (missing '("blk.0.attn_q_norm.weight" "blk.0.attn_k_norm.weight"
                     "blk.0.post_attention_norm.weight" "blk.0.post_ffw_norm.weight"))
    (let* ((spec (gemma-spec)) (table (weight-table spec)))
      (remhash missing table)
      (signals local:weights-error (with-spec-model (model "gemma3" spec table) model))))
  (let* ((spec (gemma-spec)) (table (weight-table spec)))
    (setf (car (gethash "blk.0.attn_k.weight" table)) '(1 64))
    (with-gguf-file (path (build-gguf
                           :metadata '(("general.architecture" 8 "gemma3")
                                       ("gemma3.block_count" 4 1) ("gemma3.embedding_length" 4 8)
                                       ("gemma3.attention.head_count" 4 2) ("gemma3.attention.head_count_kv" 4 1)
                                       ("gemma3.attention.key_length" 4 8)
                                       ("gemma3.attention.layer_norm_rms_epsilon" 6 1f-5))
                           :tensors (loop for name being the hash-keys of table using (hash-value entry)
                                          collect (list name (reverse (car entry)) 0 (f32-octets (cdr entry))))))
      (local:with-weights (weights path)
        (signals local:architecture-error (local:load-model weights))))))

(test gemma3-template-rendering-and-rejections
  (let ((template (make-instance 'local:gemma3-chat)))
    (is (string= (format nil "<start_of_turn>user~%Be brief.~%~%Hello<end_of_turn>~%<start_of_turn>model~%")
                 (local:render-chat template '((:role :system :content "Be brief.")
                                                (:role :user :content " Hello ")) :add-generation-prompt t)))
    (dolist (messages '(nil ((:role :assistant :content "Hi"))
                        ((:role :user :content "a") (:role :user :content "b"))
                        ((:role :user :content "a") (:role :system :content "b"))
                        ((:role :tool :content "a"))
                        ((:role :user :content ((:type :image :url "x"))))))
      (signals local:chat-template-error (local:render-chat template messages)))
    (signals local:chat-template-error (local:render-chat template *backend-messages* :tools *parser-tools*))
    (multiple-value-bind (text calls reasoning events)
        (parser-result 'local:gemma3-chat '("Hello" " world") nil)
      (is (string= "Hello world" text))
      (is (null calls)) (is (null reasoning))
      (is (string= text (event-text events))))))

(defun gemma-backend-fixture ()
  (let* ((tokens (append '("<pad>" "<eos>" "<bos>" "<unk>" "<start_of_turn>" "<end_of_turn>")
                         (loop for byte below 256 collect (format nil "<0x~2,'0X>" byte)) '("a" "</s>")))
         (spec (gemma-spec :layers 2 :dim 32 :head-dim 32 :rope-dims 32 :ffn 32 :vocab (length tokens)))
         (table (weight-table spec)))
    (maphash (lambda (name entry)
               (setf (cdr entry)
                     (loop for i below (length (cdr entry))
                           collect (if (or (string= name "token_embd.weight") (search "norm.weight" name)
                                           (and (string= name "output.weight") (= (floor i 32) 262)))
                                       1f0 0f0)))) table)
    (spec-gguf "gemma3" spec table
               `(("tokenizer.ggml.model" 8 "llama")
                 ("tokenizer.ggml.tokens" 9 (:array 8 ,tokens))
                 ("tokenizer.ggml.token_type" 9 (:array 5 ,(append '(3 3 3 2 3 3) (make-list 256 :initial-element 6) '(1 4))))
                 ("tokenizer.ggml.scores" 9 (:array 6 ,(make-list (length tokens) :initial-element 0f0)))
                 ("tokenizer.ggml.add_space_prefix" 7 nil) ("tokenizer.ggml.add_bos_token" 7 t)
                 ("tokenizer.ggml.bos_token_id" 4 2) ("tokenizer.ggml.eos_token_id" 4 1)
                 ("tokenizer.ggml.unknown_token_id" 4 3)))))

(test gemma3-client-contract-and-prefix-reuse
  (with-gguf-file (path (gemma-backend-fixture))
    (local:with-weights (weights path)
      (dolist (policy '(:serialized :per-call))
        (dolist (cache-type '(:f32 :q8))
          (let ((backend (local:make-local-backend weights :n-ctx 128 :n-batch 2
                                                           :session-policy policy :cache-type cache-type)))
            (unwind-protect
                 (progn
                   (dotimes (i 2)
                     (let* ((events nil)
                            (reply (ci:complete* backend :messages *backend-messages*
                                                        :temperature 0 :max-tokens 3
                                                        :stream (lambda (event) (push event events)))))
                       (is (string= "aaa" (ci:content-text (getf reply :content))))
                       (is (string= "aaa" (event-text (reverse events))))
                       (is (= 1 (count :done events :key (lambda (event) (getf event :type)))))
                       (when (and (= i 1) (eq policy :serialized))
                         (is (plusp (getf (getf (getf reply :meta) :usage) :cached-prompt-tokens))))))
                   (let ((reply (ci:complete* backend :messages *backend-messages* :temperature 0
                                                     :max-tokens 3 :stop '("aa"))))
                     (is (string= "" (ci:content-text (getf reply :content))))
                     (is (eq :stop (getf (getf reply :meta) :finish-reason))))
                   (dolist (options (list (list :tools *parser-tools*)
                                         (list :messages '((:role :tool :content "x")))
                                         (list :messages '((:role :user :content ((:type :image :url "x")))))))
                     (is (ci:result-error-p (apply #'ci:complete backend
                                                   (append options (list :messages *backend-messages* :max-tokens 1))))))
                   (let ((cancel (ci:make-cancel-token)))
                     (ci:cancel cancel)
                     (is (ci:result-error-p (ci:complete backend :messages *backend-messages* :cancel cancel))))
                   (is (ci:result-error-p (ci:complete backend :messages *backend-messages* :timeout 0))))
              (local:close-local-backend backend))))))))

(test gemma3-chat-bos-provenance-and-stop-ids
  (with-gguf-file (path (gemma-backend-fixture))
    (local:with-weights (weights path)
      (let* ((tokenizer (local:load-tokenizer weights)) (template (local:load-chat-template weights))
             (ids (local:encode-chat template tokenizer '((:role :user :content "<start_of_turn><bos><end_of_turn>")))))
        (is (= 1 (count 2 ids)))
        (is (= 2 (count 4 ids)))
        (is (= 1 (count 5 ids)))
        (is (equal '(5 1 263) (local:chat-template-stop-ids template tokenizer)))
        (is (not (find 263 (local:encode tokenizer "</s>" :add-bos nil))))
        (is (equal '(263) (coerce (local:encode tokenizer "</s>" :special t :add-bos nil) 'list)))
        (let ((session (local:make-session (local:load-model weights) :tokenizer tokenizer :chat-template template)))
          (unwind-protect
               (multiple-value-bind (text finish generated)
                   (local:generate session '(2 262) :sampler (script-sampler '(5)) :max-tokens 2)
                 (is (string= "" text)) (is (eq :stop finish)) (is (null generated)))
            (local:close-session session)))))))
