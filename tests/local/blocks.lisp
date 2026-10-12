(in-package #:cl-inference/local/tests)

(in-suite :cl-inference/local)

(local:define-architecture blk-llama ()
  (:hparams (n-layers) (dim) (n-heads) (n-kv-heads) (rms-epsilon)
            (rope-theta :default 10000.0) (rope-dims :default nil) (head-dim :default nil))
  (:blocks (attn local:attention) (mlp local:swiglu-mlp))
  (:model-blocks (token local:embedding) (head local:lm-head))
  (:format :gguf
    (:arch "blk-llama")
    (:rope-pairing :adjacent)
    (:hparam n-layers "~a.block_count")
    (:hparam dim "~a.embedding_length")
    (:hparam n-heads "~a.attention.head_count")
    (:hparam n-kv-heads "~a.attention.head_count_kv")
    (:hparam rms-epsilon "~a.attention.layer_norm_rms_epsilon")
    (:hparam rope-theta "~a.rope.freq_base")
    (:hparam rope-dims "~a.rope.dimension_count")
    (:hparam head-dim "~a.attention.key_length")
    (:tensor attn.norm "blk.~d.attn_norm.weight")
    (:tensor attn.q "blk.~d.attn_q.weight")
    (:tensor attn.k "blk.~d.attn_k.weight")
    (:tensor attn.v "blk.~d.attn_v.weight")
    (:tensor attn.output "blk.~d.attn_output.weight")
    (:tensor attn.rope-freqs "rope_freqs.weight" :optional t)
    (:tensor mlp.norm "blk.~d.ffn_norm.weight")
    (:tensor mlp.gate "blk.~d.ffn_gate.weight")
    (:tensor mlp.up "blk.~d.ffn_up.weight")
    (:tensor mlp.down "blk.~d.ffn_down.weight")
    (:tensor token.embd "token_embd.weight")
    (:tensor head.norm "output_norm.weight")
    (:tensor head.output "output.weight" :fallback token.embd)))

(local:define-architecture blk-neox (blk-llama)
  (:format :gguf (:arch "blk-neox") (:rope-pairing :neox)))

(local:define-architecture blk-qwen (blk-neox)
  (:blocks (attn local:attention :qk-norm rmsnorm))
  (:format :gguf
    (:arch "blk-qwen")
    (:tensor attn.q-norm "blk.~d.attn_q_norm.weight")
    (:tensor attn.k-norm "blk.~d.attn_k_norm.weight")))

(defun check-model (arch pairing spec &optional tokens)
  (let ((table (weight-table spec)) (tokens (or tokens '(3 0 5 2 6))))
    (with-spec-model (model arch spec table)
      (is (rows-close-p (tensor-rows (local:model-forward model tokens :all-logits t))
                        (ref-logits table spec pairing tokens))
          "~A ~S" arch spec)
      (is (rows-close-p (tensor-rows (local:model-forward model tokens))
                        (last (ref-logits table spec pairing tokens)))))))

(test blocks-match-reference-adjacent-rope
  (check-model "blk-llama" :adjacent (spec*)))

(test blocks-match-reference-neox-rope
  (check-model "blk-neox" :neox (spec*)))

(test blocks-untied-head
  (check-model "blk-llama" :adjacent (spec* :tied nil))
  (check-model "blk-llama" :adjacent (spec* :tied t)))

(test blocks-rope-frequency-factors
  (check-model "blk-llama" :adjacent (spec* :freqs t))
  (check-model "blk-neox" :neox (spec* :freqs t)))

(test blocks-partial-rope
  (check-model "blk-llama" :adjacent (spec* :rope-dims 2))
  (check-model "blk-neox" :neox (spec* :rope-dims 2)))

(test blocks-grouped-query-heads
  (check-model "blk-llama" :adjacent (spec* :heads 4 :kv 2 :head-dim 2 :rope-dims 2))
  (check-model "blk-llama" :adjacent (spec* :heads 4 :kv 4 :head-dim 2 :rope-dims 2)))

(test blocks-match-reference-f16
  (check-model "blk-llama" :adjacent (spec* :dtype :f16))
  (check-model "blk-qwen" :neox (spec* :dtype :f16 :qk-norm t)))

(test blocks-match-reference-bf16
  (check-model "blk-llama" :adjacent (spec* :dtype :bf16))
  (check-model "blk-neox" :neox (spec* :dtype :bf16 :freqs t)))

(test half-precision-encoders
  (is (= #x3c00 (f16-bits 1f0)))
  (is (= #xc000 (f16-bits -2f0)))
  (is (= #x7bff (f16-bits 65504f0)))
  (is (= 1 (f16-bits (scale-float 1f0 -24))))
  (is (= #x0400 (f16-bits (scale-float 1f0 -14))))
  (is (= #x3c00 (f16-bits (+ 1f0 (scale-float 1f0 -12)))))
  (is (= 1f0 (f16-value #x3c00)))
  (is (= (scale-float 1f0 -24) (f16-value 1)))
  (is (= #x3f80 (bf16-bits 1f0)))
  (is (= #xc000 (bf16-bits -2f0)))
  (is (= 1f0 (bf16-value #x3f80)))
  (dolist (x '(0.3f0 -0.7f0 1.5f-6 123.456f0))
    (is (= (f16-value (f16-bits (f16-value (f16-bits x)))) (f16-value (f16-bits x))))
    (is (= (bf16-value (bf16-bits (bf16-value (bf16-bits x)))) (bf16-value (bf16-bits x))))))

(test blocks-qk-norm
  (check-model "blk-qwen" :neox (spec* :qk-norm t)))

(test blocks-head-dim-defaults-to-dim-over-heads
  (let* ((spec (spec*)) (table (weight-table spec)))
    (with-gguf-file (path (build-gguf
                           :metadata `(("general.architecture" 8 "blk-llama")
                                       ("blk-llama.block_count" 4 2)
                                       ("blk-llama.embedding_length" 4 8)
                                       ("blk-llama.attention.head_count" 4 2)
                                       ("blk-llama.attention.head_count_kv" 4 1)
                                       ("blk-llama.attention.layer_norm_rms_epsilon" 6 1f-5))
                           :tensors (loop for (name . shape) in (weight-shapes spec)
                                          collect (list name (reverse shape) 0
                                                        (f32-octets (cdr (gethash name table)))))))
      (local:with-weights (weights path)
        (let ((model (local:load-model weights)))
          (is (rows-close-p (tensor-rows (local:model-forward model '(1 2) :all-logits t))
                            (ref-logits table (spec* :theta 10000f0) :adjacent '(1 2)))))))))

(defclass test-cache () ((entries :initform (make-hash-table) :reader cache-entries)))

(defmethod local:kv-cache-update ((cache test-cache) layer k v start)
  (declare (ignore start))
  (destructuring-bind (&optional old-k old-v) (gethash layer (cache-entries cache))
    (let ((keys (if old-k (ct:concatenate (list old-k k) :axis 0) k))
          (values (if old-v (ct:concatenate (list old-v v) :axis 0) v)))
      (setf (gethash layer (cache-entries cache)) (list keys values))
      (values keys values))))

(test blocks-cache-matches-a-full-pass
  (let* ((spec (spec* :heads 4 :kv 2 :head-dim 2 :rope-dims 2 :freqs t))
         (table (weight-table spec))
         (tokens '(3 0 5 2 6)))
    (with-spec-model (model "blk-neox" spec table)
      (let ((cache (make-instance 'test-cache)))
        (local:model-forward model (subseq tokens 0 3) :cache cache)
        (local:model-forward model (subseq tokens 3 4) :start 3 :cache cache)
        (is (rows-close-p (tensor-rows (local:model-forward model (subseq tokens 4) :start 4
                                                                                     :cache cache))
                          (last (ref-logits table spec :neox tokens))))))))

(test blocks-rope-is-relative
  (let* ((spec (spec*)) (table (weight-table spec)))
    (with-spec-model (model "blk-llama" spec table)
      (is (rows-close-p (tensor-rows (local:model-forward model '(1 2) :start 5 :all-logits t))
                        (tensor-rows (local:model-forward model '(1 2) :all-logits t))
                        1d-3)))))

(test layer-trace-matches-the-reference-per-layer
  (let* ((spec (spec* :layers 3 :dtype :f16)) (table (weight-table spec)) (tokens '(3 0 5 2 6)))
    (with-spec-model (model "blk-llama" spec table)
      (let ((expected nil)
            (trace (tracing-layers (local:model-forward model tokens))))
        (ref-logits table spec :adjacent tokens (lambda (xs) (push xs expected)))
        (is (= 3 (length trace)))
        (loop for tensor in trace for rows in (nreverse expected) for layer from 0
              do (is (rows-close-p (tensor-rows tensor) rows) "layer ~D" layer))))))

(test layer-trace-is-off-outside-tracing-layers
  (let* ((spec (spec*)) (table (weight-table spec)))
    (with-spec-model (model "blk-llama" spec table)
      (local:model-forward model '(1 2))
      (is (null *layer-trace*)))))
