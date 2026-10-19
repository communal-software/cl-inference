(in-package #:cl-inference/local/tests)
(in-suite :cl-inference/local)

(defun gemma-hf-data (name)
  (asdf:system-relative-pathname "cl-inference" (format nil "tests/local/data/gemma-hf-~A.json" name)))

(defun gemma-hf-name (name)
  (cond ((search "post_attention_norm" name)
         (format nil "model.layers.~A.post_attention_layernorm.weight"
                 (second (uiop:split-string name :separator "."))))
        ((search "post_ffw_norm" name)
         (format nil "model.layers.~A.post_feedforward_layernorm.weight"
                 (second (uiop:split-string name :separator "."))))
        ((search "ffn_norm" name)
         (format nil "model.layers.~A.pre_feedforward_layernorm.weight"
                 (second (uiop:split-string name :separator "."))))
        (t (hf-tensor-name name))))

(defun write-gemma-hf-model (directory spec table &rest overrides)
  (let ((config (hf-config "gemma3_text" spec)))
    (setf (ci::json-get config "hidden_activation") "gelu_pytorch_tanh"
          (ci::json-get config "query_pre_attn_scalar") (spec-get spec :head-dim)
          (ci::json-get config "sliding_window") (spec-get spec :window)
          (ci::json-get config "rope_local_base_freq") (spec-get spec :local-theta)
          (ci::json-get config "layer_types")
          (map 'vector (lambda (layer) (if (< (mod layer (spec-get spec :pattern))
                                             (1- (spec-get spec :pattern)))
                                         "sliding_attention" "full_attention"))
               (coerce (loop for i below (spec-get spec :layers) collect i) 'vector)))
    (loop for (key value) on overrides by #'cddr do (setf (ci::json-get config key) value))
    (write-hf-json directory "config.json" config))
  (write-octets
   (merge-pathnames "model.safetensors" directory)
   (build-safetensors
    (loop for name being the hash-keys of table using (hash-value entry)
          for dtype = (weight-dtype spec (car entry) name)
          for values = (if (search "norm" name) (mapcar (lambda (v) (- v 1f0)) (cdr entry)) (cdr entry))
          do (when (search "norm" name)
               (setf (cdr entry) (mapcar (lambda (v) (+ v 1f0)) (round-through dtype values))))
          collect (list (gemma-hf-name name) (car entry) (ecase dtype (:f32 "F32") (:f16 "F16") (:bf16 "BF16")) (dtype-octets dtype values)))))
  (write-hf-json directory "tokenizer.json" (cl-inference/hf:read-json-file (gemma-hf-data "tokenizer")))
  (write-hf-json directory "tokenizer_config.json"
                 (ci::json-object "bos_token" "<bos>" "eos_token" "<eos>" "unk_token" "<unk>" "pad_token" "<pad>")))

(test gemma-hf-offline-tokenizer-and-bos
  (with-hf-directory (directory)
    (let* ((spec (gemma-spec)) (table (weight-table spec)))
      (write-gemma-hf-model directory spec table)
      (local:with-weights (weights directory)
        (let ((tokenizer (local:load-tokenizer weights)))
          (check-hf-spm-cases tokenizer (cl-inference/hf:read-json-file (gemma-hf-data "expected")))
          (is (equalp #(2) (local:encode tokenizer "")))
          (is (equalp #() (local:encode tokenizer "" :add-bos nil)))
          (is (equal '("hello " "world") (local:pre-tokenize tokenizer "hello world")))
          (let ((config (ci::json-object "add_bos_token" nil)))
            (write-hf-json directory "tokenizer_config.json" config)
            (is (equalp #() (local:encode (local:load-tokenizer weights) "")))))))))

(test gemma-hf-cpu-logits-cache-and-source-norms
  (dolist (dtype '(:f32 :f16 :bf16))
    (dolist (tied '(nil t))
      (let* ((spec (gemma-spec :dtype dtype :tied tied :layers 3 :pattern 2 :window 3))
             (table (weight-table spec)) (tokens '(3 0 5 2 6)))
        (with-hf-directory (directory)
          (write-gemma-hf-model directory spec table)
          (local:with-weights (weights directory)
            (let* ((source (local:weights-tensor weights "model.norm.weight"))
                   (before (tensor-list (ct:astype source :f32)))
                   (model (local:load-model weights))
                   (session (local:make-session model :n-ctx 32 :n-batch 2)))
              (unwind-protect
                   (progn
                     (is (rows-close-p (tensor-rows (local:model-forward model tokens :all-logits t))
                                       (ref-logits table spec :neox tokens)))
                     (local:session-prefill session '(3 0 5))
                     (is (rows-close-p (tensor-rows (local:session-prefill session '(2 6)))
                                       (last (ref-logits table spec :neox tokens))))
                     (local:load-model weights)
                     (is (equal before (tensor-list (ct:astype source :f32)))))
                (local:close-session session)))))))))

(test gemma-hf-rejects-unsupported-model-settings
  (with-hf-directory (directory)
    (dolist (override (list '("hidden_activation" "silu") '("query_pre_attn_scalar" 1)
                           '("layer_types" #("unknown" "full_attention"))
                           '("_sliding_window_pattern" 1) '("final_logit_softcapping" 30)
                           '("attn_logit_softcapping" 50) '("use_bidirectional_attention" t)
                           '("rope_scaling" t) '("attention_bias" t)))
      (let* ((spec (gemma-spec)) (table (weight-table spec)))
        (apply #'write-gemma-hf-model directory spec table override)
        (local:with-weights (weights directory)
          (signals local:architecture-error (local:load-model weights)))))))

(test gemma-hf-rejects-tokenizer-pipeline-variants
  (with-hf-directory (directory)
    (let* ((spec (gemma-spec)) (table (weight-table spec)))
      (write-gemma-hf-model directory spec table)
      (dolist (change '((("model" "type") "Unigram") (("model" "byte_fallback") nil)
                        (("normalizer" "content") " ") (("pre_tokenizer" "behavior") "Isolated")
                        (("pre_tokenizer" "invert") cl:null)
                        (("decoder" "decoders" 0 "content") "_") (("padding") t)
                        (("post_processor" "special_tokens" "<bos>" "ids") #(1))
                        (("added_tokens" 0 "special") nil) (("added_tokens" 0 "normalized") t)))
        (write-hf-json directory "tokenizer.json"
                       (hf-spm-set-path (cl-inference/hf:read-json-file (gemma-hf-data "tokenizer"))
                                        (first change) (second change)))
        (local:with-weights (weights directory)
          (signals local:tokenizer-error (local:load-tokenizer weights)))))))

(defun verify-gemma-hf-reference
    (&key inference strict
          (root (asdf:system-relative-pathname "cl-inference" "reference/gemma-hf/"))
          (manifest (cl-inference/hf:read-json-file
                      (asdf:system-relative-pathname "cl-inference" "tools/gemma-hf-manifest.json"))))
  (let ((missing nil))
    (labels ((verify (path digest)
               (if (probe-file path)
                   (unless (equal digest (subseq (uiop:run-program
                                                  (list "shasum" "-a" "256" (namestring path))
                                                  :output :string) 0 64))
                     (error "Gemma reference hash mismatch: ~A" path))
                   (push path missing)))
             (files (table directory)
               (unless (and (hash-table-p table) (plusp (hash-table-count table)))
                 (error "Missing Gemma reference hashes"))
               (maphash (lambda (name digest)
                          (unless (and (stringp name) (equal name (file-namestring name)))
                            (error "Invalid Gemma manifest filename: ~S" name))
                          (verify (merge-pathnames name directory) digest)) table)))
      (unless (and (equal "google/gemma-3-270m-it" (gethash "repository" manifest))
                   (equal "ac82b4e820549b854eebf28ce6dedaf9fdfa17b3" (gethash "revision" manifest)))
        (error "Gemma artifact provenance differs"))
      (files (gethash "files" manifest) (merge-pathnames "model/" root))
      (when inference
        (files (gethash "outputs" manifest) root)
        (verify (asdf:system-relative-pathname "cl-inference" "tools/gemma-hf-reference.py")
                (gethash "generator_sha256" manifest))))
    (when (and missing strict) (error "Missing required Gemma HF inputs: ~S" missing))
    (unless missing root)))

(test gemma-hf-pinned-tokenizer-and-template
  (let ((root (verify-gemma-hf-reference :strict (equal "1" (uiop:getenv "CL_INFERENCE_GEMMA_HF_STRICT")))))
    (if root
        (local:with-weights (weights (merge-pathnames "model/" root))
          (let ((tokenizer (local:load-tokenizer weights)) (template (local:load-chat-template weights)))
            (check-hf-spm-cases tokenizer (cl-inference/hf:read-json-file (gemma-hf-data "real-expected")))
            (is (typep template 'local:gemma3-chat))
            (let* ((messages '((:role :user :content "literal <bos><eos><unk>")))
                   (ids (local:encode-chat template tokenizer messages)))
              (is (= 1 (count 2 ids)))
              (is (not (find 1 ids)))
              (is (not (find 3 ids))))))
        (if (uiop:getenv "CL_INFERENCE_GEMMA_HF_STRICT")
            (fail "Pinned Gemma HF artifacts are required")
            (skip "Pinned Gemma HF artifacts are absent")))))

(test gemma-hf-reference-hashes-and-strict-missing-inputs
  (with-hf-directory (directory)
    (let* ((root (merge-pathnames "model/" directory))
           (path (merge-pathnames "config.json" root))
           (manifest (alexandria:plist-hash-table
                      '("repository" "google/gemma-3-270m-it"
                        "revision" "ac82b4e820549b854eebf28ce6dedaf9fdfa17b3") :test 'equal))
           (files (make-hash-table :test 'equal)))
      (ensure-directories-exist path)
      (write-hf-json root "config.json" (ci::json-object))
      (setf (gethash "config.json" files)
            (subseq (uiop:run-program (list "shasum" "-a" "256" (namestring path)) :output :string) 0 64)
            (gethash "files" manifest) files)
      (is (equal directory (verify-gemma-hf-reference :root directory :manifest manifest :strict t)))
      (write-hf-json root "config.json" (ci::json-object "changed" t))
      (signals error (verify-gemma-hf-reference :root directory :manifest manifest))
      (rename-file path (merge-pathnames "missing.json" root))
      (is (null (verify-gemma-hf-reference :root directory :manifest manifest)))
      (signals error (verify-gemma-hf-reference :root directory :manifest manifest :strict t))
      (setf (gethash "revision" manifest) "wrong")
      (signals error (verify-gemma-hf-reference :root directory :manifest manifest)))))

(defun write-gemma-hf-chat-fixture (directory)
  (let* ((pipeline (cl-inference/hf:read-json-file (gemma-hf-data "tokenizer")))
         (vocab (gethash "vocab" (gethash "model" pipeline)))
         (added (coerce (gethash "added_tokens" pipeline) 'list)))
    (dolist (token '("<start_of_turn>" "<end_of_turn>"))
      (let ((id (hash-table-count vocab)))
        (setf (gethash token vocab) id)
        (push (alexandria:plist-hash-table
               (list "content" token "id" id "special" t "normalized" nil
                     "single_word" nil "lstrip" nil "rstrip" nil) :test 'equal) added)))
    (setf (gethash "added_tokens" pipeline) (coerce (nreverse added) 'vector))
    (let* ((spec (gemma-spec :vocab (hash-table-count vocab) :layers 2 :pattern 2 :window 3))
           (table (weight-table spec)))
      (write-gemma-hf-model directory spec table)
      (write-hf-json directory "tokenizer.json" pipeline))))

(test gemma-hf-rejects-malformed-shapes-and-missing-post-norms
  (with-hf-directory (directory)
    (dolist (malformation '(:missing :shape))
      (let* ((spec (gemma-spec)) (table (weight-table spec)))
        (ecase malformation
          (:missing (remhash "blk.0.post_attention_norm.weight" table))
          (:shape (setf (gethash "blk.0.post_attention_norm.weight" table) (cons '(1) '(1f0)))))
        (write-gemma-hf-model directory spec table)
        (local:with-weights (weights directory)
          (ecase malformation
            (:missing (signals local:weights-error (local:load-model weights)))
            (:shape (signals local:architecture-error (local:load-model weights)))))))))

(defun check-gemma-hf-client (maker closer)
  (dolist (policy '(:serialized :per-call))
    (with-hf-directory (directory)
      (write-gemma-hf-chat-fixture directory)
      (local:with-weights (weights directory)
        (let ((backend (funcall maker weights :session-policy policy :n-ctx 128 :n-batch 7)))
          (unwind-protect
               (let* ((messages '((:role :system :content "brief") (:role :user :content "hello <bos>")))
                      (events nil)
                      (first (ci:complete backend :messages messages :max-tokens 3 :temperature 0
                                          :stream (lambda (event) (push event events))))
                      (second (ci:complete backend :messages messages :max-tokens 3 :temperature 0)))
                 (is (not (ci:result-error-p first)))
                 (is (equal (getf (second first) :content) (getf (second second) :content)))
                 (is (string= (ci:content-text (getf (second first) :content)) (event-text (reverse events))))
                 (is (= 1 (count :done events :key (lambda (event) (getf event :type)))))
                 (let ((cancel (ci:make-cancel-token)))
                   (ci:cancel cancel)
                   (is (equal '(:error :cancelled) (ci:complete backend :messages messages :cancel cancel))))
                 (is (equal '(:error :timeout) (ci:complete backend :messages messages :timeout 0)))
                 (is (not (ci:result-error-p (ci:complete backend :messages messages :max-tokens 1 :temperature 0)))))
            (funcall closer backend)))))))

(test gemma-hf-cpu-client-prefix-streaming-and-cancellation
  (check-gemma-hf-client #'local:make-local-backend #'local:close-local-backend))
