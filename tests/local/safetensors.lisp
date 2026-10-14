(in-package #:cl-inference/local/tests)

(in-suite :cl-inference/local)

(defun build-safetensors (tensors &key metadata header header-length trailing)
  (let ((object (ci::json-object)) (payload (make-buffer)) (out (make-buffer)))
    (when metadata (setf (ci::json-get object "__metadata__") metadata))
    (loop for (name shape dtype octets) in tensors
          for start = (length payload)
          do (put-octets payload octets)
             (setf (ci::json-get object name)
                   (ci::json-object "dtype" dtype "shape" (coerce shape 'vector)
                                    "data_offsets" (vector start (length payload)))))
    (let ((text (or header (com.inuoe.jzon:stringify object))))
      (setf text (concatenate 'string text (make-string (mod (- (length text)) 8)
                                                       :initial-element #\Space)))
      (let ((octets (babel:string-to-octets text :encoding :utf-8)))
        (put-uint out (or header-length (length octets)) 8)
        (put-octets out octets)))
    (put-octets out payload)
    (when trailing (put-octets out trailing))
    out))

(defmacro with-hf-directory ((directory) &body body)
  `(uiop:with-temporary-file (:pathname token :prefix "cl-inference-hf-" :stream stream)
     (finish-output stream)
     :close-stream
     (let ((,directory (uiop:ensure-directory-pathname (concatenate 'string (namestring token) "-bundle/"))))
       (ensure-directories-exist (merge-pathnames "model.safetensors" ,directory))
       (unwind-protect (progn ,@body)
         (uiop:delete-directory-tree ,directory :validate t)))))

(defun write-hf-json (directory filename value)
  (write-octets (merge-pathnames filename directory)
               (babel:string-to-octets (if (stringp value) value (com.inuoe.jzon:stringify value))
                                       :encoding :utf-8)))

(defun open-bad-safetensors (octets)
  (with-gguf-file (path octets) (local:open-weights path :format :safetensors)))

(defun small-safetensors ()
  (build-safetensors `(("matrix" (2 3) "F32" ,(f32-octets '(0f0 1f0 2f0 3f0 4f0 5f0)))
                       ("scalar" () "F16" ,(u16-octets '(#x4000)))
                       ("empty" (0 3) "BF16" #()))))

(test safetensors-detection-views-and-lifetime
  (with-gguf-file (path (small-safetensors))
    (let ((before (read-octets path)) (w (local:open-weights path)))
      (unwind-protect
           (progn
             (is (eq :safetensors (local:weights-format w)))
             (is (equal '("matrix" "scalar" "empty") (local:weights-tensor-names w)))
             (is (equal '(2 3) (local:tensor-info-shape (local:weights-tensor-info w "matrix"))))
             (is (= 24 (local:tensor-info-byte-count (local:weights-tensor-info w "matrix"))))
             (is (= 5f0 (ct:tref (local:weights-tensor w "matrix") 1 2)))
             (is (= 2f0 (ct:tref (ct:astype (local:weights-tensor w "scalar") :f32))))
             (is (= 0 (ct:tensor-size (local:weights-tensor w "empty"))))
             (is (= 0 (ct:tensor-size (local:weights-tensor-bytes w "empty"))))
             (is (null (local:weights-architecture-name w)))
             (signals local:unknown-architecture (local:load-architecture w))
             (signals local:weights-error (local:weights-tensor w "absent"))
             (setf (ct:tref (local:weights-tensor w "matrix") 1 2) 9f0))
        (local:close-weights w))
      (is (equalp before (read-octets path)))
      (local:close-weights w)
      (signals local:weights-closed (local:weights-tensor w "matrix"))
      (signals local:weights-closed (local:weights-tensor-bytes w "matrix"))
      (signals local:weights-closed (local:weights-metadata w "model_type"))
      (signals local:weights-closed (local:weights-tensor-names w))
      (signals local:weights-closed (local:weights-architecture-name w)))))

(test safetensors-all-typed-dtypes
  (dolist (spec '(("F32" :f32 4 1065353216 1f0) ("F64" :f64 8 4607182418800017408 1d0)
                  ("F16" :f16 2 15360 1f0) ("BF16" :bf16 2 16256 1f0)
                  ("I8" :s8 1 255 -1) ("U8" :u8 1 255 255)
                  ("I16" :s16 2 65535 -1) ("U16" :u16 2 65535 65535)
                  ("I32" :s32 4 4294967295 -1) ("U32" :u32 4 4294967295 4294967295)
                  ("I64" :s64 8 18446744073709551615 -1)
                  ("U64" :u64 8 18446744073709551615 18446744073709551615)))
    (destructuring-bind (name dtype bytes bits value) spec
      (let ((payload (make-buffer)))
        (put-uint payload bits bytes)
        (with-gguf-file (path (build-safetensors `(("w" (1) ,name ,payload))))
          (local:with-weights (w path :format :safetensors)
            (let ((tensor (local:weights-tensor w "w")))
              (is (eq dtype (ct:tensor-dtype tensor)))
              (is (= value (ct:tref (if (member dtype '(:f16 :bf16)) (ct:astype tensor :f32) tensor) 0)))
              (is (= bytes (local:tensor-info-byte-count (local:weights-tensor-info w "w")))))))))))

(test safetensors-raw-only-dtypes
  (dolist (dtype '("BOOL" "F8_E4M3" "F8_E5M2" "F8_E8M0"))
    (with-gguf-file (path (build-safetensors `(("w" (2) ,dtype #(0 1)))))
      (local:with-weights (w path)
        (signals local:unsupported-weights-type (local:weights-tensor w "w"))
        (is (equal '(0 1) (tensor-list (local:weights-tensor-bytes w "w"))))))))

(test safetensors-config-and-header-metadata
  (with-hf-directory (directory)
    (write-hf-json directory "config.json"
                   "{\"model_type\":\"qwen3\",\"flag\":false,\"none\":null,\"items\":[1,2],\"quantization\":{\"bits\":4}}")
    (write-octets (merge-pathnames "model.safetensors" directory)
                  (build-safetensors nil :metadata (ci::json-object "format" "mlx")))
    (local:with-weights (w directory)
      (is (equal '("model_type" "flag" "none" "items" "quantization" "safetensors.__metadata__")
                 (local:weights-metadata-keys w)))
      (is (equal '(nil t) (multiple-value-list (local:weights-metadata w "flag" :absent))))
      (is (equal '(cl:null t) (multiple-value-list (local:weights-metadata w "none" :absent))))
      (is (equal '(:absent nil) (multiple-value-list (local:weights-metadata w "missing" :absent))))
      (is (equalp #(1 2) (local:weights-metadata w "items")))
      (is (= 4 (gethash "bits" (local:weights-metadata w "quantization"))))
      (is (equal "qwen3" (local:weights-architecture-name w)))
      (is (equal "mlx" (gethash "format" (gethash "model.safetensors"
                                                   (local:weights-metadata w "safetensors.__metadata__"))))))))

(test safetensors-rejects-bad-headers
  (dolist (octets (list #() #(1 2 3)
                        (build-safetensors nil :header-length 100000001)
                        (build-safetensors nil :header-length 999)
                        (build-safetensors nil :header "[]")
                        (build-safetensors nil :header " {}")
                        (build-safetensors nil :header "{\"x\":{},\"x\":{}}")
                        (build-safetensors nil :header "{\"__metadata__\":{\"a\":1}}")
                        (build-safetensors nil :header "{\"__metadata__\":{\"a\":\"b\",\"a\":\"c\"}}")
                        (build-safetensors nil :header "{} {}")
                        (build-safetensors nil :header "{broken}")
                        (build-safetensors nil :header "{\"a\":{\"dtype\":\"F4\",\"shape\":[1],\"data_offsets\":[0,1]}}")))
    (signals st:safetensors-error (open-bad-safetensors octets)))
  (let ((octets (build-safetensors nil)))
    (setf (aref octets 9) 255)
    (signals st:safetensors-error (open-bad-safetensors octets)))
  (let ((text (concatenate 'string "{\"a\":" (make-string 130 :initial-element #\[)
                           "0" (make-string 130 :initial-element #\]) "}")))
    (signals st:safetensors-error (open-bad-safetensors (build-safetensors nil :header text)))))

(test safetensors-rejects-bad-payloads
  (dolist (entry '("{\"dtype\":\"F32\",\"shape\":[-1],\"data_offsets\":[0,4]}"
                   "{\"dtype\":\"F32\",\"shape\":[1.5],\"data_offsets\":[0,4]}"
                   "{\"dtype\":\"F32\",\"shape\":null,\"data_offsets\":[0,4]}"
                   "{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,4]}"
                   "{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[4,0]}"
                   "{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[1,5]}"
                   "{\"dtype\":\"WHAT\",\"shape\":[1],\"data_offsets\":[0,4]}"))
    (signals st:safetensors-error
      (open-bad-safetensors (build-safetensors nil :header (format nil "{\"w\":~A}" entry)
                                               :trailing #(0 0 0 0)))))
  (signals st:safetensors-error
    (open-bad-safetensors (build-safetensors nil :trailing #(0))))
  (signals st:safetensors-error
    (open-bad-safetensors (subseq (small-safetensors) 0 (1- (length (small-safetensors))))))
  (dolist (offset '(0 8))
    (signals st:safetensors-error
      (open-bad-safetensors
       (build-safetensors nil
                          :header (format nil "{\"a\":{\"dtype\":\"U8\",\"shape\":[4],\"data_offsets\":[0,4]},\"b\":{\"dtype\":\"U8\",\"shape\":[4],\"data_offsets\":[~D,~D]}}"
                                          offset (+ offset 4))
                          :trailing (make-array (+ offset 4) :initial-element 0))))))

(defun write-hf-index (directory pairs)
  (let ((map (ci::json-object)))
    (dolist (pair pairs) (setf (ci::json-get map (car pair)) (cdr pair)))
    (write-hf-json directory "model.safetensors.index.json" (ci::json-object "weight_map" map))))

(test safetensors-sharded-directory-and-index
  (with-hf-directory (directory)
    (write-octets (merge-pathnames "a.safetensors" directory)
                  (build-safetensors '(("z" (1) "U8" #(7)) ("a" (1) "U8" #(8)))))
    (write-octets (merge-pathnames "b.safetensors" directory)
                  (build-safetensors '(("b" (1) "U8" #(9)))))
    (write-hf-index directory '(("b" . "b.safetensors") ("a" . "a.safetensors") ("z" . "a.safetensors")))
    (write-octets (merge-pathnames "model.safetensors" directory) (build-safetensors nil))
    (dolist (path (list directory (merge-pathnames "model.safetensors.index.json" directory)))
      (local:with-weights (w path)
        (is (= 2 (length (st::weights-shards w))))
        (is (equal '("z" "a" "b") (local:weights-tensor-names w)))
        (is (= 9 (ct:tref (local:weights-tensor w "b") 0)))))
    (dolist (pairs '((("z" . "absent.safetensors"))
                     (("z" . "a.safetensors"))
                     (("z" . "a.safetensors") ("a" . "a.safetensors") ("missing" . "b.safetensors"))
                     (("z" . "b.safetensors") ("a" . "a.safetensors") ("b" . "a.safetensors"))
                     (("z" . "../escape.safetensors")) (("z" . "/tmp/escape.safetensors"))))
      (write-hf-index directory pairs)
      (signals st:safetensors-error (local:open-weights directory)))
    (write-hf-index directory '(("z" . "a.safetensors") ("a" . "a.safetensors") ("b" . "b.safetensors")))
    (write-octets (merge-pathnames "b.safetensors" directory)
                  (build-safetensors '(("z" (1) "U8" #(9)))))
    (signals st:safetensors-error (local:open-weights directory))))

(test safetensors-malformed-config-and-index
  (with-hf-directory (directory)
    (signals st:safetensors-error (local:open-weights directory :format :safetensors))
    (write-octets (merge-pathnames "model.safetensors" directory) (build-safetensors nil))
    (dolist (config '("[]" "{\"x\":1,\"x\":2}" "{\"safetensors.__metadata__\":{}}"))
      (write-hf-json directory "config.json" config)
      (signals st:safetensors-error (local:open-weights directory)))
    (write-hf-json directory "config.json" "{}")
    (dolist (index '("{}" "{\"weight_map\":[]}" "{\"weight_map\":{}}"
                    "{\"weight_map\":{\"w\":4}}" "{\"weight_map\":{\"w\":\"a\",\"w\":\"b\"}}"))
      (write-hf-json directory "model.safetensors.index.json" index)
      (signals st:safetensors-error (local:open-weights directory)))))

(defun hf-tensor-name (name)
  (or (cdr (assoc name '(("token_embd.weight" . "model.embed_tokens.weight")
                         ("output_norm.weight" . "model.norm.weight")
                         ("output.weight" . "lm_head.weight")) :test #'equal))
      (let* ((parts (uiop:split-string name :separator "."))
             (suffix (cdr (assoc (third parts)
                                '(("attn_norm" . "input_layernorm")
                                  ("attn_q" . "self_attn.q_proj") ("attn_k" . "self_attn.k_proj")
                                  ("attn_v" . "self_attn.v_proj") ("attn_output" . "self_attn.o_proj")
                                  ("attn_q_norm" . "self_attn.q_norm") ("attn_k_norm" . "self_attn.k_norm")
                                  ("ffn_norm" . "post_attention_layernorm")
                                  ("ffn_gate" . "mlp.gate_proj") ("ffn_up" . "mlp.up_proj")
                                  ("ffn_down" . "mlp.down_proj")) :test #'equal))))
        (unless suffix (error "No HF fixture mapping for ~A" name))
        (format nil "model.layers.~A.~A.~A" (second parts) suffix (fourth parts)))))

(defun hf-config (arch spec &rest overrides)
  (let ((config (ci::json-object
                 "model_type" arch "hidden_size" (spec-get spec :dim)
                 "num_hidden_layers" (spec-get spec :layers)
                 "num_attention_heads" (spec-get spec :heads) "num_key_value_heads" (spec-get spec :kv)
                 "head_dim" (spec-get spec :head-dim) "rms_norm_eps" (spec-get spec :epsilon)
                 "rope_theta" (spec-get spec :theta) "max_position_embeddings" 128
                 "hidden_act" "silu" "tie_word_embeddings" (if (spec-get spec :tied) t nil)
                 "rope_scaling" 'cl:null)))
    (loop for (key value) on overrides by #'cddr do (setf (ci::json-get config key) value))
    config))

(defun hf-model-octets (spec table)
  (build-safetensors
   (loop for name being the hash-keys of table using (hash-value entry)
         unless (equal name "rope_freqs.weight")
           collect (list (hf-tensor-name name) (car entry)
                         (ecase (weight-dtype spec (car entry)) (:f32 "F32") (:f16 "F16") (:bf16 "BF16"))
                         (dtype-octets (weight-dtype spec (car entry)) (cdr entry))))))

(defun write-hf-model (directory arch spec table &rest config-overrides)
  (write-octets (merge-pathnames "model.safetensors" directory) (hf-model-octets spec table))
  (write-hf-json directory "config.json" (apply #'hf-config arch spec config-overrides)))

(test safetensors-models-match-independent-reference
  (dolist (arch '("llama" "qwen3"))
    (dolist (dtype '(:f32 :f16 :bf16))
      (dolist (tied '(nil t))
        (let* ((spec (spec* :qk-norm (equal arch "qwen3") :head-dim 8 :rope-dims 8
                            :dtype dtype :tied tied))
               (table (weight-table spec)) (tokens '(3 0 5 2 6)))
          (with-hf-directory (directory)
            (write-hf-model directory arch spec table)
            (local:with-weights (w directory)
              (let* ((model (local:load-model w)) (session (local:make-session model :n-batch 2)))
                (is (rows-close-p (tensor-rows (local:model-forward model tokens :all-logits t))
                                  (ref-logits table spec :neox tokens)) "~A ~A tied=~A" arch dtype tied)
                (is (rows-close-p (tensor-rows (local:model-forward model tokens))
                                  (last (ref-logits table spec :neox tokens))))
                (local:session-prefill session (subseq tokens 0 3))
                (is (rows-close-p (tensor-rows (local:session-prefill session (subseq tokens 3)))
                                  (last (ref-logits table spec :neox tokens))))))))))))

(test safetensors-hf-defaults-and-qwen3-required-norms
  (let* ((spec (spec* :qk-norm t :tied t)) (table (weight-table spec)))
    (with-hf-directory (directory)
      (write-hf-model directory "qwen3" spec table "head_dim" 'cl:null)
      (local:with-weights (w directory)
        (let ((model (local:load-model w)))
          (is (rows-close-p (tensor-rows (local:model-forward model '(1 2) :all-logits t))
                            (ref-logits table spec :neox '(1 2))))))
      (dolist (name '("blk.0.attn_q_norm.weight" "blk.0.attn_k_norm.weight"))
        (let ((entry (gethash name table)))
          (remhash name table)
          (write-hf-model directory "qwen3" spec table)
          (local:with-weights (w directory) (signals local:weights-error (local:load-model w)))
          (setf (gethash name table) entry))))))

(defun llama3-scaling ()
  (ci::json-object "rope_type" "llama3" "factor" 8 "low_freq_factor" 1
                   "high_freq_factor" 4 "original_max_position_embeddings" 128))

(test safetensors-llama3-scaling-matches-fixed-oracle
  (let* ((spec (spec* :head-dim 8 :rope-dims 8 :theta 10000f0 :freqs t))
         (table (weight-table spec)) (tokens '(3 0 5 2 6))
         ;; Frequency factors from Transformers' _compute_llama3_parameters.
         (expected '(1f0 2.3391168f0 8f0 8f0)))
    (setf (gethash "rope_freqs.weight" table) (cons '(4) expected))
    (with-hf-directory (directory)
      (write-hf-model directory "llama" spec table "rope_scaling" (llama3-scaling))
      (local:with-weights (w directory)
        (let* ((model (local:load-model w))
               (attention (first (first (local::model-layers model)))))
          (is (every (lambda (a b) (< (abs (- a b)) 1f-6))
                     (tensor-list (local::attn-rope-freqs attention)) expected))
          (is (rows-close-p (tensor-rows (local:model-forward model tokens :all-logits t))
                            (ref-logits table spec :neox tokens))))))))

(test safetensors-cpu-rejects-unsupported-configurations
  (let* ((spec (spec*)) (table (weight-table spec)))
    (with-hf-directory (directory)
      (dolist (override (list (list "rope_scaling" (ci::json-object "rope_type" "yarn"))
                             (list "rope_scaling" (ci::json-object "rope_type" "llama3" "factor" 0))
                             (list "rope_parameters" (ci::json-object "rope_type" "default"))
                             (list "quantization" (ci::json-object "bits" 4 "group_size" 64))
                             (list "quantization_config" (ci::json-object "bits" 4))
                             (list "attention_bias" t) (list "mlp_bias" t)
                             (list "hidden_act" "gelu") (list "partial_rotary_factor" 0.5)
                             (list "use_sliding_window" t) (list "sliding_window" 32)
                             (list "head_dim" 3) (list "rms_norm_eps" 0)
                             (list "num_key_value_heads" 3)))
        (apply #'write-hf-model directory "llama" spec table override)
        (local:with-weights (w directory)
          (signals local:architecture-error (local:load-model w))))
      (remhash "output.weight" table)
      (write-hf-model directory "llama" spec table)
      (local:with-weights (w directory) (signals local:architecture-error (local:load-model w)))
      (setf (gethash "blk.0.attn_q.bias" table) (cons '(8) (make-list 8 :initial-element 0f0)))
      (write-hf-model directory "llama" spec table "tie_word_embeddings" t)
      (local:with-weights (w directory) (signals local:architecture-error (local:load-model w))))))

(test safetensors-preserves-mlx-packed-storage
  (with-hf-directory (directory)
    (write-octets (merge-pathnames "model.safetensors" directory)
                  (build-safetensors `(("model.layers.0.self_attn.q_proj.weight" (1 2) "U32" #(255 0 0 0 254 0 0 0))
                                       ("model.layers.0.self_attn.q_proj.scales" (1 1) "F16" ,(u16-octets '(#x3c00)))
                                       ("model.layers.0.self_attn.q_proj.biases" (1 1) "F16" ,(u16-octets '(0))))))
    (write-hf-json directory "config.json"
                   (hf-config "qwen3" (spec*) "quantization" (ci::json-object "bits" 4 "group_size" 64)))
    (local:with-weights (w directory)
      (is (= 4 (gethash "bits" (local:weights-metadata w "quantization"))))
      (is (= 255 (ct:tref (local:weights-tensor w "model.layers.0.self_attn.q_proj.weight") 0 0)))
      (is (eq :u32 (local:tensor-info-format-type
                     (local:weights-tensor-info w "model.layers.0.self_attn.q_proj.weight"))))
      (is (= 2 (local:tensor-info-byte-count
                 (local:weights-tensor-info w "model.layers.0.self_attn.q_proj.scales"))))
      (signals local:architecture-error (local:load-model w)))))

(defun observe-file-mappings (thunk &key fail-first-unmap)
  (let* ((map (fdefinition 'cl-inference/mapping:map-file))
         (unmap (fdefinition 'cl-inference/mapping:unmap-file))
         (opened nil) (released nil))
    (unwind-protect
         (progn
           (setf (fdefinition 'cl-inference/mapping:map-file)
                 (lambda (path minimum fail)
                   (multiple-value-bind (pointer size) (funcall map path minimum fail)
                     (push (cffi:pointer-address pointer) opened)
                     (values pointer size)))
                 (fdefinition 'cl-inference/mapping:unmap-file)
                 (lambda (pointer size fail)
                   (funcall unmap pointer size fail)
                   (push (cffi:pointer-address pointer) released)
                   (when (and fail-first-unmap (= (length released) 1))
                     (funcall fail "Injected unmap failure"))))
           (funcall thunk))
      (setf (fdefinition 'cl-inference/mapping:map-file) map
            (fdefinition 'cl-inference/mapping:unmap-file) unmap))
    (is (equal (sort (copy-list opened) #'<) (sort (copy-list released) #'<)))
    (length opened)))

(test safetensors-releases-mappings-on-every-exit
  (with-hf-directory (directory)
    (write-octets (merge-pathnames "a.safetensors" directory)
                  (build-safetensors '(("a" (1) "U8" #(1)))))
    (write-octets (merge-pathnames "b.safetensors" directory)
                  (build-safetensors '(("b" (1) "U8" #(2)))))
    (write-hf-index directory '(("a" . "a.safetensors") ("b" . "b.safetensors")))
    (is (= 2 (observe-file-mappings
              (lambda ()
                (let ((w (local:open-weights directory)))
                  (signals st:safetensors-error (local:close-weights w))
                  (local:close-weights w))) :fail-first-unmap t)))
    (is (= 2 (observe-file-mappings
              (lambda ()
                (signals simple-error
                  (local:with-weights (w directory) (error "Body failed")))))))
    (write-octets (merge-pathnames "b.safetensors" directory)
                  (build-safetensors '(("b" (1) "U8" #(2))) :trailing #(9)))
    (is (= 2 (observe-file-mappings
              (lambda () (signals st:safetensors-error (local:open-weights directory))))))
    (write-octets (merge-pathnames "b.safetensors" directory)
                  (build-safetensors '(("b" (1) "U8" #(2)))))
    (write-hf-index directory '(("a" . "b.safetensors") ("b" . "a.safetensors")))
    (is (= 2 (observe-file-mappings
              (lambda () (signals st:safetensors-error (local:open-weights directory))))))))

(test safetensors-rejects-existing-shards-outside-directory
  (with-hf-directory (directory)
    (with-hf-directory (outside)
      (let ((escaped (merge-pathnames "outside.safetensors" outside)))
        (write-octets escaped (build-safetensors '(("w" (1) "U8" #(1)))))
        (write-hf-index directory `(("w" . ,(enough-namestring escaped directory))))
        (signals st:safetensors-error (local:open-weights directory))
        (uiop:run-program (list "ln" "-s" (namestring escaped)
                                (namestring (merge-pathnames "link.safetensors" directory))))
        (write-hf-index directory '(("w" . "link.safetensors")))
        (signals st:safetensors-error (local:open-weights directory))))))
