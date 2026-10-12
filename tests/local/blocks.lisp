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

;;; A model spec is a plist; the same weights feed the GGUF file and the reference.

(defun spec-get (spec key) (getf spec key))

(defun weight-shapes (spec)
  (destructuring-bind (&key dim heads kv head-dim layers vocab ffn rope-dims qk-norm tied freqs
                         &allow-other-keys)
      spec
    (append
     `(("token_embd.weight" ,vocab ,dim) ("output_norm.weight" ,dim))
     (unless tied `(("output.weight" ,vocab ,dim)))
     (when freqs `(("rope_freqs.weight" ,(floor rope-dims 2))))
     (loop for l below layers
           append (flet ((n (fmt) (format nil fmt l)))
                    `((,(n "blk.~d.attn_norm.weight") ,dim)
                      (,(n "blk.~d.attn_q.weight") ,(* heads head-dim) ,dim)
                      (,(n "blk.~d.attn_k.weight") ,(* kv head-dim) ,dim)
                      (,(n "blk.~d.attn_v.weight") ,(* kv head-dim) ,dim)
                      (,(n "blk.~d.attn_output.weight") ,dim ,(* heads head-dim))
                      ,@(when qk-norm `((,(n "blk.~d.attn_q_norm.weight") ,head-dim)
                                        (,(n "blk.~d.attn_k_norm.weight") ,head-dim)))
                      (,(n "blk.~d.ffn_norm.weight") ,dim)
                      (,(n "blk.~d.ffn_gate.weight") ,ffn ,dim)
                      (,(n "blk.~d.ffn_up.weight") ,ffn ,dim)
                      (,(n "blk.~d.ffn_down.weight") ,dim ,ffn)))))))

(defun weight-values (name shape)
  "Deterministic single floats; norms and frequency factors sit near 1."
  (let* ((seed (reduce #'+ (map 'list #'char-code name)))
         (count (reduce #'* shape))
         (near-one (or (search "norm" name) (search "freqs" name))))
    (loop for i below count
          collect (float (+ (if near-one 1.0 0.0) (* (if near-one 0.3 0.5) (sin (+ seed (* 1.3 i)))))
                         1f0))))

(defun weight-table (spec)
  (let ((table (make-hash-table :test 'equal)))
    (loop for (name . shape) in (weight-shapes spec)
          do (setf (gethash name table) (cons shape (weight-values name shape))))
    table))

(defun spec-gguf (arch spec table)
  (destructuring-bind (&key dim heads kv head-dim layers rope-dims epsilon theta &allow-other-keys)
      spec
    (build-gguf
     :metadata `(("general.architecture" 8 ,arch)
                 (,(format nil "~a.block_count" arch) 4 ,layers)
                 (,(format nil "~a.embedding_length" arch) 4 ,dim)
                 (,(format nil "~a.attention.head_count" arch) 4 ,heads)
                 (,(format nil "~a.attention.head_count_kv" arch) 4 ,kv)
                 (,(format nil "~a.attention.key_length" arch) 4 ,head-dim)
                 (,(format nil "~a.attention.layer_norm_rms_epsilon" arch) 6 ,epsilon)
                 (,(format nil "~a.rope.freq_base" arch) 6 ,theta)
                 (,(format nil "~a.rope.dimension_count" arch) 4 ,rope-dims))
     :tensors (loop for (name . shape) in (weight-shapes spec)
                    collect (list name (reverse shape) 0
                                  (f32-octets (cdr (gethash name table))))))))

;;; Double-precision reference: vectors are lists, matrices are lists of rows.

(defun ref-weight (table name)
  "Rows of a rank-2 weight, or the list of a rank-1 one."
  (destructuring-bind (shape . values) (gethash name table)
    (if (rest shape)
        (loop for r below (first shape)
              collect (mapcar (lambda (v) (coerce v 'double-float))
                              (subseq values (* r (second shape)) (* (1+ r) (second shape)))))
        (mapcar (lambda (v) (coerce v 'double-float)) values))))

(defun ref-dot (a b) (reduce #'+ (mapcar #'* a b)))
(defun ref-matvec (rows x) (mapcar (lambda (row) (ref-dot row x)) rows))
(defun ref-rmsnorm (x w eps)
  (let ((scale (/ 1d0 (sqrt (+ eps (/ (ref-dot x x) (length x)))))))
    (mapcar (lambda (v weight) (* v scale weight)) x w)))
(defun ref-silu (x) (/ x (+ 1d0 (exp (- x)))))

(defun ref-rope (vec position rope-dims theta factors pairing)
  (let ((result (copy-list vec)) (half (floor rope-dims 2)))
    (dotimes (i half result)
      (let* ((frequency (/ (expt (coerce theta 'double-float) (/ (* -2d0 i) rope-dims))
                           (if factors (nth i factors) 1d0)))
             (angle (* position frequency))
             (a (if (eq pairing :adjacent) (* 2 i) i))
             (b (if (eq pairing :adjacent) (1+ (* 2 i)) (+ i half)))
             (x0 (nth a vec)) (x1 (nth b vec)))
        (setf (nth a result) (- (* x0 (cos angle)) (* x1 (sin angle)))
              (nth b result) (+ (* x0 (sin angle)) (* x1 (cos angle))))))))

(defun ref-split (list size) (loop for i from 0 below (length list) by size
                                   collect (subseq list i (+ i size))))

(defun ref-attention (table spec pairing layer xs)
  (destructuring-bind (&key heads kv head-dim rope-dims epsilon theta qk-norm freqs &allow-other-keys)
      spec
    (flet ((w (suffix) (ref-weight table (format nil "blk.~d.~a" layer suffix))))
      (let* ((normed (mapcar (lambda (x) (ref-rmsnorm x (w "attn_norm.weight") epsilon)) xs))
             (factors (when freqs (ref-weight table "rope_freqs.weight")))
             (qs (loop for h in normed for pos from 0
                       collect (mapcar (lambda (head)
                                         (ref-rope (if qk-norm
                                                       (ref-rmsnorm head (w "attn_q_norm.weight") epsilon)
                                                       head)
                                                   pos rope-dims theta factors pairing))
                                       (ref-split (ref-matvec (w "attn_q.weight") h) head-dim))))
             (ks (loop for h in normed for pos from 0
                       collect (mapcar (lambda (head)
                                         (ref-rope (if qk-norm
                                                       (ref-rmsnorm head (w "attn_k_norm.weight") epsilon)
                                                       head)
                                                   pos rope-dims theta factors pairing))
                                       (ref-split (ref-matvec (w "attn_k.weight") h) head-dim))))
             (vs (loop for h in normed
                       collect (ref-split (ref-matvec (w "attn_v.weight") h) head-dim)))
             (group (floor heads kv)))
        (loop for qi below (length xs)
              collect
              (let ((merged
                      (loop for head below heads
                            append
                            (let* ((kvh (floor head group))
                                   (scores (loop for kj to qi
                                                 collect (/ (ref-dot (nth head (nth qi qs))
                                                                     (nth kvh (nth kj ks)))
                                                            (sqrt (coerce head-dim 'double-float)))))
                                   (top (reduce #'max scores))
                                   (exps (mapcar (lambda (s) (exp (- s top))) scores))
                                   (total (reduce #'+ exps)))
                              (loop for d below head-dim
                                    collect (/ (loop for e in exps for kj from 0
                                                     sum (* e (nth d (nth kvh (nth kj vs)))))
                                               total))))))
                (ref-matvec (w "attn_output.weight") merged)))))))

(defun ref-mlp (table spec layer xs)
  (let ((eps (spec-get spec :epsilon)))
    (flet ((w (suffix) (ref-weight table (format nil "blk.~d.~a" layer suffix))))
      (loop for x in xs
            collect (let* ((h (ref-rmsnorm x (w "ffn_norm.weight") eps))
                           (gate (ref-matvec (w "ffn_gate.weight") h))
                           (up (ref-matvec (w "ffn_up.weight") h)))
                      (ref-matvec (w "ffn_down.weight")
                                  (mapcar (lambda (g u) (* (ref-silu g) u)) gate up)))))))

(defun ref-logits (table spec pairing tokens)
  "Logits for every position of TOKENS."
  (let* ((embd (ref-weight table "token_embd.weight"))
         (xs (mapcar (lambda (token) (copy-list (nth token embd))) tokens)))
    (dotimes (layer (spec-get spec :layers))
      (setf xs (mapcar #'(lambda (x d) (mapcar #'+ x d)) xs
                       (ref-attention table spec pairing layer xs)))
      (setf xs (mapcar #'(lambda (x d) (mapcar #'+ x d)) xs (ref-mlp table spec layer xs))))
    (let ((output (if (gethash "output.weight" table) (ref-weight table "output.weight") embd)))
      (mapcar (lambda (x)
                (ref-matvec output (ref-rmsnorm x (ref-weight table "output_norm.weight")
                                                (spec-get spec :epsilon))))
              xs))))

(defun tensor-rows (tensor)
  (loop for i below (aref (ct:tensor-shape tensor) 0)
        collect (loop for j below (aref (ct:tensor-shape tensor) 1) collect (ct:tref tensor i j))))

(defun rows-close-p (actual expected &optional (tolerance 1d-4))
  (and (= (length actual) (length expected))
       (every (lambda (a e) (every (lambda (x y) (< (abs (- x y)) tolerance)) a e)) actual expected)))

(defparameter *base-spec*
  '(:dim 8 :heads 2 :kv 1 :head-dim 4 :layers 2 :vocab 7 :ffn 10 :rope-dims 4
    :epsilon 1f-5 :theta 500000f0))

(defun spec* (&rest overrides)
  (let ((spec (copy-list *base-spec*)))
    (loop for (key value) on overrides by #'cddr do (setf (getf spec key) value))
    spec))

(defmacro with-spec-model ((model arch spec table) &body body)
  `(with-gguf-file (path (spec-gguf ,arch ,spec ,table))
     (local:with-weights (weights path)
       (let ((,model (local:load-model weights)))
         ,@body))))

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

(defvar *around-calls* 0)

(defmethod local:block-forward :around ((block local:attention) x &key &allow-other-keys)
  (incf *around-calls*)
  (call-next-method))

(test blocks-around-methods-wrap-block-forward
  (let ((*around-calls* 0))
    (check-model "blk-llama" :adjacent (spec*))
    (is (plusp *around-calls*))))
