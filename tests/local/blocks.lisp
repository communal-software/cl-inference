(in-package #:cl-inference/local/tests)

(in-suite :cl-inference/local)

(local:define-architecture blk-llama (local:llama)
  (:format :gguf (:arch "blk-llama"))
  (:format :safetensors (:arch nil)))

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
  (dolist (path '(:auto :encoded :tiled))
    (let ((local::*half-linear-path* path))
      (check-model "blk-llama" :adjacent (spec* :dtype :f16))
      (check-model "blk-qwen" :neox (spec* :dtype :f16 :qk-norm t)))))

(test blocks-match-reference-bf16
  (dolist (path '(:auto :encoded :tiled))
    (let ((local::*half-linear-path* path))
      (check-model "blk-llama" :adjacent (spec* :dtype :bf16))
      (check-model "blk-neox" :neox (spec* :dtype :bf16 :freqs t)))))

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

(defun half-weight (rows columns dtype)
  "A (ROWS COLUMNS) DTYPE weight of small values, and its f32 values."
  (let* ((bits (ecase dtype (:f16 #'f16-bits) (:bf16 #'bf16-bits)))
         (value (ecase dtype (:f16 #'f16-value) (:bf16 #'bf16-value)))
         (raw (make-array (* rows columns) :element-type '(unsigned-byte 16)))
         (floats (make-array (* rows columns) :element-type 'single-float)))
    (dotimes (i (* rows columns))
      (setf (aref raw i) (funcall bits (* 0.01f0 (- (mod (* i 37) 101) 50)))
            (aref floats i) (funcall value (aref raw i))))
    (values (ct:make-tensor-view raw (list rows columns) :dtype dtype)
            (ct:make-tensor-view floats (list rows columns)))))

(defun half-activations (rows columns)
  (ct:from-data (loop for r below rows collect
                 (loop for c below columns collect
                   (* 0.1f0 (- (mod (+ (* r 5) (* c 3)) 11) 5))))))

(defun check-half-linear (x weight decoded)
  (let* ((values (mapcar (lambda (row) (mapcar (lambda (v) (coerce v 'double-float)) row))
                        (tensor-rows decoded)))
         (expected (mapcar (lambda (row)
                             (ref-matvec values (mapcar (lambda (v) (coerce v 'double-float)) row)))
                           (tensor-rows x))))
    (dolist (path '(:encoded :tiled :auto))
      (let* ((local::*half-linear-path* path) (result (local:linear x weight)))
        (is (eq :f32 (ct:tensor-dtype result)))
        (is (rows-close-p (tensor-rows result) expected)
            "~S ~S ~S" path (ct:tensor-dtype weight) (ct:tensor-shape weight))))))

(test linear-over-half-precision-matches-reference
  (dolist (dtype '(:f16 :bf16))
    (dolist (rows '(1 7 255 256 257 512))
      (dolist (columns '(1 7 33 65))
        (multiple-value-bind (weight decoded) (half-weight rows columns dtype)
          (dolist (count '(1 6 46))
            (check-half-linear (half-activations count columns) weight decoded)))))))

(test half-linear-offsets-and-strides
  (dolist (dtype '(:f16 :bf16))
    (multiple-value-bind (weight decoded) (half-weight 260 70 dtype)
      (let ((x (ct:slice (half-activations 8 70) :selectors '((1 7) (2 67)))))
        (check-half-linear x
                           (ct:slice weight :selectors '((1 258) (2 67)))
                           (ct:slice decoded :selectors '((1 258) (2 67)))))
      (dolist (selectors '(((1 258) (2 67 2)) ((257 0 -1) (66 1 -2))))
        (let ((view (ct:slice weight :selectors selectors))
              (reference (ct:slice decoded :selectors selectors)))
          (check-half-linear (half-activations 6 (aref (ct:tensor-shape view) 1)) view reference)))
      (check-half-linear (half-activations 6 260) (ct:transpose weight) (ct:transpose decoded))
      (check-half-linear (half-activations 6 65)
                         (ct:make-tensor-view (ct:tensor-storage weight) '(257 65)
                                             :dtype dtype :strides '(0 0))
                         (ct:make-tensor-view (ct:tensor-storage decoded) '(257 65)
                                             :strides '(0 0))))))

(test half-linear-empty-and-invalid
  (dolist (dtype '(:f16 :bf16))
    (dolist (shape '((0 7 3) (6 0 3) (6 7 0)))
      (destructuring-bind (count rows columns) shape
        (dolist (path '(:encoded :tiled :auto))
          (let* ((local::*half-linear-path* path)
                 (result (local:linear (ct:make-tensor (list count columns))
                                       (ct:make-tensor (list rows columns) :dtype dtype))))
            (is (equalp (vector count rows) (ct:tensor-shape result)))
            (is (eq :f32 (ct:tensor-dtype result)))
            (is (every #'zerop (ct:tensor-storage result)))))))
    (dolist (path '(:encoded :tiled :auto))
      (let ((local::*half-linear-path* path))
        (signals error (local:linear (ct:make-tensor '(2 4))
                                     (ct:make-tensor '(3 5) :dtype dtype)))
        (signals error (local:linear (ct:make-tensor '(2 4) :dtype :f64)
                                     (ct:make-tensor '(3 4) :dtype dtype)))))))

(defmacro with-function-wrapper ((name original wrapper) &body body)
  `(let ((,original (fdefinition ,name)))
     (unwind-protect
          (progn (setf (fdefinition ,name) ,wrapper) ,@body)
       (setf (fdefinition ,name) ,original))))

(test half-linear-mapped-storage-and-scratch
  (dolist (dtype '(:f16 :bf16))
    (multiple-value-bind (weight decoded) (half-weight 260 70 dtype)
      (declare (ignore weight))
      (with-gguf-file (path (build-gguf :tensors
                             (list (list "w" '(70 260) (dtype-id dtype)
                                         (dtype-octets dtype (apply #'append (tensor-rows decoded)))))))
        (local:with-weights (weights path)
          (let* ((mapped (ct:slice (local:weights-tensor weights "w")
                                   :selectors '((1 258) (2 67))))
                 (x (half-activations 6 65))
                 (function (ecase dtype (:f16 'trivial-simd:f16-matmul!)
                                       (:bf16 'trivial-simd:bf16-matmul!)))
                 (calls 0))
            (with-function-wrapper
                (function original
                 (lambda (out activations storage n m k &rest options)
                   (incf calls)
                   (is (eq storage (ct:tensor-storage mapped)))
                   (is (= (ct:tensor-offset mapped) (getf options :weights-start)))
                   (is (= 70 (getf options :weights-stride)))
                   (apply original out activations storage n m k options)))
              (with-function-wrapper
                  ('trivial-simd:convert! converter
                   (lambda (&rest arguments)
                     (declare (ignore arguments))
                     (error "Encoded linear unexpectedly converts storage")))
                (with-function-wrapper
                    ('trivial-simd:nd-convert! nd-converter
                     (lambda (&rest arguments)
                       (declare (ignore arguments))
                       (error "Encoded linear unexpectedly converts strided storage")))
                  (let ((local::*half-linear-path* :encoded)) (local:linear x mapped))
                  (with-function-wrapper
                      ('local::half-linear-profile-p profile (constantly t))
                    (let ((local::*half-linear-path* :auto)
                          (ct:*executor* (lambda (&rest arguments)
                                          (declare (ignore arguments)) (error "Unexpected scheduling"))))
                      (with-function-wrapper
                          ('trivial-simd:encoded-matmul-native-p capability (constantly t))
                        (local:linear (ct:slice x :selectors '((0 1) :all)) mapped)))))))
            (is (= 2 calls))
            (with-function-wrapper
                ('trivial-simd:nd-convert! original
                 (lambda (destination input shape &rest options)
                   (is (<= (length destination) (* local::+tile-rows+ 65)))
                   (is (eq input (ct:tensor-storage mapped)))
                   (apply original destination input shape options)))
              (let ((local::*half-linear-path* :tiled)) (local:linear x mapped)))
            (check-half-linear x mapped (ct:slice decoded :selectors '((1 258) (2 67))))))))))

(test half-linear-profile-and-fallback
  (multiple-value-bind (weight decoded) (half-weight 257 33 :f16)
    (let ((x (half-activations 6 33)))
      (with-function-wrapper
          ('local::half-linear-profile-p original (constantly t))
        (with-function-wrapper
            ('trivial-simd:encoded-matmul-native-p capability (constantly t))
            (let ((local::*half-linear-path* :auto)
                (local::*half-linear-thresholds* '((:f16 . 6) (:bf16 . 1))))
            (is (eq :encoded (local::half-linear-path x weight)))
            (is (eq :tiled (local::half-linear-path (half-activations 7 33) weight))))))
      (with-function-wrapper
          ('trivial-simd:encoded-matmul-native-p capability (constantly nil))
        (let ((local::*half-linear-path* :auto))
          (is (eq :tiled (local::half-linear-path x weight)))))
      (with-function-wrapper
          ('local::half-linear-profile-p profile (constantly nil))
        (let ((local::*half-linear-path* :auto))
          (is (eq :tiled (local::half-linear-path x weight)))))
      (let ((trivial-simd::*native-encoded-matmul-p* nil))
        (check-half-linear x weight decoded))
      (let ((trivial-simd::*backend* :lisp))
        (check-half-linear x weight decoded))
      (let ((ct:*executor* (lambda (count work) (funcall work 0 count))))
        (is (eq :tiled (local::half-linear-path x weight)))
        (check-half-linear x weight decoded)))))

(test half-linear-single-row-executor-dispatch
  (dolist (dtype '(:f16 :bf16))
    (multiple-value-bind (weight decoded) (half-weight 257 33 dtype)
      (declare (ignore decoded))
      (with-function-wrapper
          ('local::half-linear-profile-p profile (constantly t))
        (with-function-wrapper
            ('trivial-simd:encoded-matmul-native-p capability (constantly t))
          (let ((local::*half-linear-path* :auto)
                (ct:*executor* (lambda (&rest arguments)
                                (declare (ignore arguments)) (error "Unexpected scheduling"))))
            (dolist (count '(0 1 2 18 19 21 22))
              (let ((x (ct:make-tensor (list count 33))))
                (is (eq (if (= count 1) :encoded :tiled) (local::half-linear-path x weight))))))
          (let* ((local::*half-linear-path* :auto)
                 (threshold (cdr (assoc dtype local::*half-linear-thresholds*))))
            (is (eq :encoded (local::half-linear-path (half-activations threshold 33) weight)))
            (is (eq :tiled (local::half-linear-path (half-activations (1+ threshold) 33) weight)))))))))

(test half-linear-single-row-executor-views
  (dolist (dtype '(:f16 :bf16))
    (multiple-value-bind (weight decoded) (half-weight 260 70 dtype)
      (with-function-wrapper
          ('local::half-linear-profile-p profile (constantly t))
        (let ((ct:*executor* (lambda (&rest arguments)
                              (declare (ignore arguments)) (error "Unexpected scheduling"))))
          (dolist (selectors '(:all ((1 258) (2 67)) ((1 258) (2 67 2))
                              ((257 0 -1) (66 1 -2))))
            (let* ((view (if (eq selectors :all) weight (ct:slice weight :selectors selectors)))
                   (reference (if (eq selectors :all) decoded (ct:slice decoded :selectors selectors)))
                   (columns (aref (ct:tensor-shape view) 1))
                   (x (ct:slice (half-activations 3 (* 2 columns))
                                :selectors (list '(1 2) (list 0 (* 2 columns) 2)))))
              (with-function-wrapper
                  ('trivial-simd:encoded-matmul-native-p capability (constantly t))
                (let ((local::*half-linear-path* :auto))
                  (is (eq (if (or (eq selectors :all) (equal selectors '((1 258) (2 67))))
                              :encoded :tiled)
                          (local::half-linear-path x view)))))
              (check-half-linear x view reference)))
          (check-half-linear (half-activations 1 260) (ct:transpose weight) (ct:transpose decoded))
          (let ((x (half-activations 1 70)) (local::*half-linear-path* :auto))
            (with-function-wrapper
                ('trivial-simd:encoded-matmul-native-p capability (constantly nil))
              (is (eq :tiled (local::half-linear-path x weight))))
            (with-function-wrapper
                ('local::half-linear-profile-p original (constantly nil))
              (is (eq :tiled (local::half-linear-path x weight))))
            (let ((ct:*executor* :invalid) (local::*half-linear-path* :encoded))
              (signals error (local:linear x weight)))))))))

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
