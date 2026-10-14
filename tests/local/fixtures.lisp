(in-package #:cl-inference/local/tests)

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

(defun weight-dtype (spec shape &optional name)
  "Matrices use :tensor-dtypes overrides or :dtype; vectors stay f32."
  (if (rest shape)
      (or (cdr (assoc name (spec-get spec :tensor-dtypes) :test #'equal)) (spec-get spec :dtype) :f32)
      :f32))

(defun dtype-bits (dtype)
  (ecase dtype
    (:f16 (values #'f16-bits #'f16-value))
    (:bf16 (values #'bf16-bits #'bf16-value))))

(defun q8-0-blocks (values)
  "Quantize VALUES in blocks of 32 the way ggml does. Returns the stored f16 scales, and the
quantized integers of each block."
  (loop for start from 0 below (length values) by 32
        for block = (subseq values start (+ start 32))
        for scale = (/ (reduce #'max block :key #'abs) 127)
        for stored = (f16-bits scale)
        collect (cons stored
                      (let ((divisor (f16-value stored)))
                        (mapcar (lambda (value) (if (zerop divisor) 0 (round value scale)))
                                block)))))

(defun round-through (dtype values)
  "VALUES as DTYPE stores them, so the reference sees what the file holds."
  (case dtype
    (:f32 values)
    ((:q4_0 :q4_k :q6_k) (nth-value 1 (pack-test-values dtype values)))
    (:q8_0 (loop for (stored . quantized) in (q8-0-blocks values)
                 append (mapcar (lambda (q) (* q (f16-value stored))) quantized)))
    (t (multiple-value-bind (encode decode) (dtype-bits dtype)
         (mapcar (lambda (value) (funcall decode (funcall encode value))) values)))))

(defun q8-0-octets (values)
  (let ((buffer (make-buffer)))
    (loop for (stored . quantized) in (q8-0-blocks values)
          do (put-uint buffer stored 2)
             (dolist (q quantized) (put-uint buffer (ldb (byte 8 0) q) 1)))
    buffer))

(defun dtype-octets (dtype values)
  (case dtype
    (:f32 (f32-octets values))
    ((:q4_0 :q4_k :q6_k) (pack-test-values dtype values))
    (:q8_0 (q8-0-octets values))
    (t (u16-octets (mapcar (nth-value 0 (dtype-bits dtype)) values)))))

(defun dtype-id (dtype)
  (ecase dtype (:f32 0) (:f16 1) (:q4_0 2) (:q8_0 8) (:q4_k 12) (:q6_k 14) (:bf16 30)))

(defun weight-table (spec)
  (let ((table (make-hash-table :test 'equal)))
    (loop for (name . shape) in (weight-shapes spec)
          do (setf (gethash name table)
                   (cons shape (round-through (weight-dtype spec shape name)
                                              (weight-values name shape)))))
    table))

(defun spec-gguf (arch spec table &optional extra-metadata)
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
                 (,(format nil "~a.rope.dimension_count" arch) 4 ,rope-dims)
                 ,@extra-metadata)
     :tensors (loop for (name . shape) in (weight-shapes spec)
                    for dtype = (weight-dtype spec shape name)
                    collect (list name (reverse shape) (dtype-id dtype)
                                  (dtype-octets dtype (cdr (gethash name table))))))))

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

(defun ref-logits (table spec pairing tokens &optional on-layer)
  "Logits for every position of TOKENS. ON-LAYER, when given, is called with the residual
stream after each layer."
  (let* ((embd (ref-weight table "token_embd.weight"))
         (xs (mapcar (lambda (token) (copy-list (nth token embd))) tokens)))
    (dotimes (layer (spec-get spec :layers))
      (setf xs (mapcar #'(lambda (x d) (mapcar #'+ x d)) xs
                       (ref-attention table spec pairing layer xs)))
      (setf xs (mapcar #'(lambda (x d) (mapcar #'+ x d)) xs (ref-mlp table spec layer xs)))
      (when on-layer (funcall on-layer xs)))
    (let ((output (if (gethash "output.weight" table) (ref-weight table "output.weight") embd)))
      (mapcar (lambda (x)
                (ref-matvec output (ref-rmsnorm x (ref-weight table "output_norm.weight")
                                                (spec-get spec :epsilon))))
              xs))))

;;; Residual-stream traces

(defvar *layer-trace* nil
  "When bound to a list, collects the residual stream after each layer, newest first.")

(defmethod local:layer-forward :around (architecture blocks x &key &allow-other-keys)
  (let ((result (call-next-method)))
    (when *layer-trace* (push result (car *layer-trace*)))
    result))

(defmacro tracing-layers (&body body)
  "Run BODY and return the residual tensor after each layer it ran, oldest first."
  `(let ((*layer-trace* (list nil)))
     ,@body
     (reverse (car *layer-trace*))))

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

