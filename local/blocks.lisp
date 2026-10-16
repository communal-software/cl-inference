(in-package #:cl-inference/local)

;;; Blocks are bound to their weights at load time. A block's tensor roles are its slot name
;;; plus a suffix: slot ATTN reads roles ATTN.Q, ATTN.K, and so on.

(defclass model-block ()
  ((slot :initarg :slot :reader block-slot)
   (layer :initarg :layer :initform nil :reader block-layer)))

(defgeneric bind-block (block architecture weights)
  (:documentation "Read BLOCK's tensors and hparams from WEIGHTS into its slots."))

(defgeneric block-forward (block input &key &allow-other-keys)
  (:documentation "Run BLOCK on INPUT. Per-layer blocks return a residual delta."))

(defgeneric kv-cache-update (cache layer k v start)
  (:documentation "Store K and V of the new positions at START for LAYER in CACHE and return
all keys and values so far, as (values keys values). Shapes are (positions kv-heads head-dim)."))

(defun block-tensor (block architecture weights suffix)
  "The tensor for role <slot>.<SUFFIX>, or NIL when it is optional and absent."
  (architecture-tensor architecture weights
                       (intern (format nil "~A.~A" (symbol-name (block-slot block))
                                       (string-upcase suffix))
                               '#:cl-inference/local)
                       (block-layer block)))

(defun hparam-or (architecture name default)
  (handler-case (or (hparam architecture name) default)
    (architecture-error () default)))

(defun as-f32 (tensor)
  "TENSOR as f32; for small tensors, since it converts the whole thing."
  (if (eq (ct:tensor-dtype tensor) :f32) tensor (ct:astype tensor :f32)))

(defconstant +tile-rows+ 256
  "Weight rows converted to f32 at a time by TILED-LINEAR.")

(defvar *half-linear-path* :auto
  "Internal path override for tests and whole-call measurements.")

(defparameter *half-linear-thresholds* '((:f16 . 18) (:bf16 . 21))
  "Contiguous counts with at least 10% savings across the measured shapes.")

;; Preserve profile replacement in compiled ECL fixtures (#54).
(declaim (notinline half-linear-profile-p))

(defun half-linear-profile-p ()
  (and (eq (uiop:operating-system) :macosx)
       (equal (lisp-implementation-type) "SBCL")
       (equal (lisp-implementation-version) "2.6.8")
       (equal (machine-type) "ARM64")
       (equal (machine-version) "Apple M1")
       (eq (trivial-simd:backend) :native)))

(defun half-linear-direct-weight-p (weight)
  (let ((shape (ct:tensor-shape weight)) (strides (ct:tensor-strides weight)))
    (and (or (= (aref shape 1) 1) (= (aref strides 1) 1))
         (or (= (aref shape 0) 1) (>= (aref strides 0) (aref shape 1))))))

(defun half-linear-path (x weight)
  (unless (and (= (ct:tensor-rank x) 2) (= (ct:tensor-rank weight) 2)
               (eq (ct:tensor-dtype x) :f32)
               (= (aref (ct:tensor-shape x) 1) (aref (ct:tensor-shape weight) 1)))
    (error "Half-precision linear requires matching f32 activation and weight matrices"))
  (ecase *half-linear-path*
    ((:encoded :tiled) *half-linear-path*)
    (:auto
     (if (and (half-linear-profile-p)
              (trivial-simd:encoded-matmul-native-p)
              (if ct:*executor*
                  (and (= (aref (ct:tensor-shape x) 0) 1)
                       (half-linear-direct-weight-p weight))
                  ;; TODO: serial packing can dominate; profile layout-aware cutoffs (#59).
                  (<= (aref (ct:tensor-shape x) 0)
                      (cdr (assoc (ct:tensor-dtype weight) *half-linear-thresholds*)))))
         :encoded :tiled))))

(defun tiled-linear (x weight)
  "X · WEIGHTᵀ with at most +TILE-ROWS+ weight rows converted to f32."
  (destructuring-bind (rows columns) (coerce (ct:tensor-shape weight) 'list)
    (let* ((tile-rows (min +tile-rows+ rows))
           (tile (make-array (* tile-rows columns) :element-type 'single-float))
           (out (ct:make-tensor (list (aref (ct:tensor-shape x) 0) rows)))
           (encoding (ct:tensor-dtype weight))
           (strides (ct:tensor-strides weight)))
      (when (or (zerop (ct:tensor-size out)) (zerop columns))
        (return-from tiled-linear out))
      (loop for start from 0 below rows by tile-rows
            for end = (min rows (+ start tile-rows))
            for offset = (+ (ct:tensor-offset weight) (* start (aref strides 0)))
            do (if (ct:contiguous-p weight)
                   (trivial-simd:convert! tile (ct:tensor-storage weight)
                                         :input-encoding encoding :input-start offset
                                         :end (* (- end start) columns))
                   (trivial-simd:nd-convert! tile (ct:tensor-storage weight)
                                            (list (- end start) columns)
                                            :input-encoding encoding :input-start offset
                                            :input-strides strides))
               (ct:matmul! (ct:slice out :selectors (list :all (list start end)))
                           x (ct:transpose (ct:make-tensor-view tile (list (- end start) columns)))))
      out)))

(defun linear (x weight)
  "X · WEIGHTᵀ for a WEIGHT of shape (out in). Every projection goes through here."
  (case (ct:tensor-dtype weight)
    ((:f16 :bf16)
     (ecase (half-linear-path x weight)
       (:encoded (ct:matmul x (ct:transpose weight)))
       (:tiled (tiled-linear x weight))))
    (t (ct:matmul x (ct:transpose weight)))))

(defun apply-rmsnorm (x weight epsilon)
  (ct:rmsnorm x :weights (as-f32 weight) :epsilon epsilon))

;;; Standard blocks

(defclass rmsnorm (model-block)
  ((weight :accessor rmsnorm-weight) (epsilon :accessor rmsnorm-epsilon)))

(defmethod bind-block ((block rmsnorm) architecture weights)
  (setf (rmsnorm-weight block) (block-tensor block architecture weights "norm")
        (rmsnorm-epsilon block) (hparam architecture 'rms-epsilon)))

(defmethod block-forward ((block rmsnorm) x &key &allow-other-keys)
  (apply-rmsnorm x (rmsnorm-weight block) (rmsnorm-epsilon block)))

(defclass embedding (model-block)
  ((table :accessor embedding-table)))

(defmethod bind-block ((block embedding) architecture weights)
  (setf (embedding-table block) (block-tensor block architecture weights "embd")))

(defmethod block-forward ((block embedding) tokens &key &allow-other-keys)
  "Rows of the table for the integer tensor TOKENS, as f32."
  (as-f32 (ct:take (embedding-table block) tokens)))

(defclass swiglu-mlp (model-block)
  ((norm :accessor mlp-norm) (gate :accessor mlp-gate) (up :accessor mlp-up)
   (down :accessor mlp-down) (epsilon :accessor mlp-epsilon)))

(defmethod bind-block ((block swiglu-mlp) architecture weights)
  (setf (mlp-norm block) (block-tensor block architecture weights "norm")
        (mlp-gate block) (block-tensor block architecture weights "gate")
        (mlp-up block) (block-tensor block architecture weights "up")
        (mlp-down block) (block-tensor block architecture weights "down")
        (mlp-epsilon block) (hparam architecture 'rms-epsilon)))

(defmethod block-forward ((block swiglu-mlp) x &key &allow-other-keys)
  (let ((h (apply-rmsnorm x (mlp-norm block) (mlp-epsilon block))))
    (linear (ct:multiply (ct:silu (linear h (mlp-gate block))) (linear h (mlp-up block)))
            (mlp-down block))))

(defclass lm-head (model-block)
  ((norm :accessor head-norm) (output :accessor head-output) (epsilon :accessor head-epsilon)))

(defmethod bind-block ((block lm-head) architecture weights)
  (setf (head-norm block) (block-tensor block architecture weights "norm")
        (head-output block) (block-tensor block architecture weights "output")
        (head-epsilon block) (hparam architecture 'rms-epsilon)))

(defmethod block-forward ((block lm-head) x &key last-only &allow-other-keys)
  "Logits for every row of X, or for its last row when LAST-ONLY."
  (let* ((rows (aref (ct:tensor-shape x) 0))
         (x (if last-only (ct:slice x :selectors (list (list (1- rows) rows))) x)))
    (linear (apply-rmsnorm x (head-norm block) (head-epsilon block)) (head-output block))))

;;; RoPE

(defun rope-tables (start count rope-dims theta factors)
  "Cosine and sine tables of shape (COUNT 1 ROPE-DIMS/2) for positions START.., with
frequency i = THETA^(-2i/ROPE-DIMS) / FACTORS[i]."
  (let* ((half (floor rope-dims 2))
         (cos (ct:make-tensor (list count 1 half)))
         (sin (ct:make-tensor (list count 1 half))))
    (dotimes (i half)
      (let ((frequency (/ (expt (coerce theta 'double-float) (/ (* -2d0 i) rope-dims))
                          (if factors (coerce (ct:tref factors i) 'double-float) 1d0))))
        (dotimes (pos count)
          (let ((angle (* (+ start pos) frequency)))
            (setf (ct:tref cos pos 0 i) (coerce (cl:cos angle) 'single-float)
                  (ct:tref sin pos 0 i) (coerce (cl:sin angle) 'single-float))))))
    (values cos sin)))

(defun rotate-pair (x0 x1 cos sin)
  (values (ct:subtract (ct:multiply x0 cos) (ct:multiply x1 sin))
          (ct:add (ct:multiply x0 sin) (ct:multiply x1 cos))))

(defun apply-rope (x cos sin pairing rope-dims)
  "Rotate the first ROPE-DIMS of the last axis of X (positions heads head-dim)."
  (let* ((count (aref (ct:tensor-shape x) 0))
         (heads (aref (ct:tensor-shape x) 1))
         (head-dim (aref (ct:tensor-shape x) 2))
         (half (floor rope-dims 2))
         (rotated (ct:slice x :selectors (list :all :all (list 0 rope-dims))))
         (result
           (ecase pairing
             (:adjacent
              (let ((pairs (ct:reshape rotated (list count heads half 2))))
                (multiple-value-bind (e0 e1)
                    (rotate-pair (ct:slice pairs :selectors (list :all :all :all 0))
                                 (ct:slice pairs :selectors (list :all :all :all 1)) cos sin)
                  (ct:reshape (ct:stack (list e0 e1) :axis 3) (list count heads rope-dims)))))
             (:neox
              (multiple-value-bind (e0 e1)
                  (rotate-pair (ct:slice rotated :selectors (list :all :all (list 0 half)))
                               (ct:slice rotated :selectors (list :all :all (list half rope-dims)))
                               cos sin)
                (ct:concatenate (list e0 e1) :axis 2))))))
    (if (= rope-dims head-dim)
        result
        (ct:concatenate (list result (ct:slice x :selectors (list :all :all (list rope-dims nil))))
                        :axis 2))))

;;; Attention

(defclass attention (model-block)
  ((qk-norm :initarg :qk-norm :initform nil :reader attention-qk-norm)
   (norm :accessor attn-norm) (q :accessor attn-q) (k :accessor attn-k) (v :accessor attn-v)
   (output :accessor attn-output) (q-norm :accessor attn-q-norm) (k-norm :accessor attn-k-norm)
   (rope-freqs :accessor attn-rope-freqs)
   (n-heads :accessor attn-n-heads) (n-kv-heads :accessor attn-n-kv-heads)
   (head-dim :accessor attn-head-dim) (rope-dims :accessor attn-rope-dims)
   (rope-theta :accessor attn-rope-theta) (rope-pairing :accessor attn-rope-pairing)
   (epsilon :accessor attn-epsilon)))

(defun llama3-rope-factors (dims theta scaling)
  (let ((mode (and (hash-table-p scaling)
                   (or (gethash "rope_type" scaling) (gethash "type" scaling)))))
    (when (equal mode "default") (return-from llama3-rope-factors nil))
    (unless (equal mode "llama3")
      (architecture-fail 'architecture-error "Unsupported RoPE scaling ~S" mode))
    (let ((factor (gethash "factor" scaling)) (low (gethash "low_freq_factor" scaling))
          (high (gethash "high_freq_factor" scaling))
          (context (gethash "original_max_position_embeddings" scaling)))
      (unless (and (realp factor) (>= factor 1) (realp low) (plusp low)
                   (realp high) (> high low) (integerp context) (plusp context))
        (architecture-fail 'architecture-error "Invalid Llama 3 RoPE scaling parameters"))
      (let ((factors (ct:make-tensor (list (floor dims 2)))))
        (dotimes (i (floor dims 2) factors)
          (let* ((frequency (expt (coerce theta 'double-float) (/ (* -2d0 i) dims)))
                 (wavelength (/ (* 2 pi) frequency))
                 (smooth (/ (- (/ context wavelength) low) (- high low)))
                 (multiplier (cond ((< wavelength (/ context high)) 1d0)
                                   ((> wavelength (/ context low)) (/ 1d0 factor))
                                   (t (+ (/ (- 1 smooth) factor) smooth)))))
            (setf (ct:tref factors i) (coerce (/ 1d0 multiplier) 'single-float))))))))

(defmethod bind-block ((block attention) architecture weights)
  (flet ((tensor (suffix) (block-tensor block architecture weights suffix)))
    (let* ((n-heads (hparam architecture 'n-heads))
           (head-dim (hparam-or architecture 'head-dim
                                (floor (hparam architecture 'dim) n-heads))))
      (setf (attn-norm block) (tensor "norm") (attn-q block) (tensor "q")
            (attn-k block) (tensor "k") (attn-v block) (tensor "v")
            (attn-output block) (tensor "output") (attn-rope-freqs block) (tensor "rope-freqs")
            (attn-n-heads block) n-heads
            (attn-n-kv-heads block) (hparam-or architecture 'n-kv-heads n-heads)
            (attn-head-dim block) head-dim
            (attn-rope-dims block) (hparam-or architecture 'rope-dims head-dim)
            (attn-rope-theta block) (hparam-or architecture 'rope-theta 10000.0)
            (attn-rope-pairing block) (architecture-format-option
                                       architecture (architecture-format architecture)
                                       :rope-pairing :adjacent)
            (attn-epsilon block) (hparam architecture 'rms-epsilon))
      (let ((scaling (hparam-or architecture 'rope-scaling nil)))
        (when scaling
          (setf (attn-rope-freqs block)
                (llama3-rope-factors (attn-rope-dims block) (attn-rope-theta block) scaling))))
      (when (attention-qk-norm block)
        (setf (attn-q-norm block) (tensor "q-norm") (attn-k-norm block) (tensor "k-norm"))))))

(defun causal-mask (count key-count query-start key-start)
  "A (COUNT KEY-COUNT) :u8 mask, 1 where the key's position is not after the query's; NIL
when nothing is masked."
  (let ((mask (ct:make-tensor (list count key-count) :dtype :u8)) (masked nil))
    (dotimes (i count)
      (dotimes (j key-count)
        (if (<= (+ key-start j) (+ query-start i))
            (setf (ct:tref mask i j) 1)
            (setf masked t))))
    (and masked mask)))

(defun call-with-head-ranges (count function)
  (let ((executor ct:*executor*)
        (backend trivial-simd::*backend*)
        (fma trivial-simd::*fma-mode*))
    (unless (or (null executor) (functionp executor))
      (error "Executor must be a function or NIL"))
    (flet ((work (start end)
             (unless (and (integerp start) (integerp end) (<= 0 start end count))
               (error "Invalid attention head range [~S,~S) for ~D heads" start end count))
             (let ((ct:*executor* nil)
                   (trivial-simd::*backend* backend)
                   (trivial-simd::*fma-mode* fma))
               (funcall function start end))))
      (cond ((zerop count) nil)
            ((or (null executor) (= count 1)) (work 0 count))
            (t (funcall executor count #'work))))))

(defun attend (q keys values mask n-heads n-kv-heads head-dim)
  "Grouped-query attention of Q (positions heads head-dim) over KEYS and VALUES."
  (let ((group (floor n-heads n-kv-heads))
        (scale (/ 1.0 (sqrt (coerce head-dim 'single-float))))
        (results (make-array n-heads)))
    (call-with-head-ranges
     n-heads
     (lambda (start end)
       (loop for head from start below end
             for kv = (floor head group)
             for scores = (ct:multiply
                           (ct:matmul (ct:slice q :selectors (list :all head))
                                      (ct:transpose (ct:slice keys :selectors (list :all kv))))
                           scale)
             for masked = (if mask (ct:select mask scores -1f30) scores)
             do (setf (aref results head)
                      (ct:matmul (ct:softmax masked) (ct:slice values :selectors (list :all kv)))))))
    (ct:concatenate (coerce results 'list) :axis 1)))

(defmethod block-forward ((block attention) x &key (start 0) cache &allow-other-keys)
  "Residual delta of attention over X. With CACHE, K and V go through KV-CACHE-UPDATE and X
holds the positions from START; without, X attends only to itself."
  (let* ((count (aref (ct:tensor-shape x) 0))
         (n-heads (attn-n-heads block)) (n-kv-heads (attn-n-kv-heads block))
         (head-dim (attn-head-dim block))
         (h (apply-rmsnorm x (attn-norm block) (attn-epsilon block)))
         (q (ct:reshape (linear h (attn-q block)) (list count n-heads head-dim)))
         (k (ct:reshape (linear h (attn-k block)) (list count n-kv-heads head-dim)))
         (v (ct:reshape (linear h (attn-v block)) (list count n-kv-heads head-dim))))
    (when (attention-qk-norm block)
      (setf q (apply-rmsnorm q (attn-q-norm block) (attn-epsilon block))
            k (apply-rmsnorm k (attn-k-norm block) (attn-epsilon block))))
    (multiple-value-bind (cos sin)
        (rope-tables start count (attn-rope-dims block) (attn-rope-theta block)
                     (and (attn-rope-freqs block) (as-f32 (attn-rope-freqs block))))
      (setf q (apply-rope q cos sin (attn-rope-pairing block) (attn-rope-dims block))
            k (apply-rope k cos sin (attn-rope-pairing block) (attn-rope-dims block))))
    (multiple-value-bind (keys values key-start)
        (if cache
            (multiple-value-bind (keys values) (kv-cache-update cache (block-layer block) k v start)
              (values keys values 0))
            (values k v start))
      (linear (attend q keys values
                      (causal-mask count (aref (ct:tensor-shape keys) 0) start key-start)
                      n-heads n-kv-heads head-dim)
              (attn-output block)))))

;;; Models

(defstruct (model (:constructor make-model (architecture layers model-blocks)))
  architecture
  layers          ; one list of blocks per layer
  model-blocks)   ; blocks that run once

(defun make-block (architecture spec slot layer weights)
  (destructuring-bind (class &rest initargs) spec
    (let ((block (apply #'make-instance class :slot slot :layer layer initargs)))
      (bind-block block architecture weights)
      block)))

(defgeneric validate-model-weights (weights architecture)
  (:method (weights architecture) (declare (ignore weights architecture)) nil))

(defun load-model (weights)
  "Build the architecture WEIGHTS declares and bind its blocks. Valid until WEIGHTS closes."
  (let* ((architecture (load-architecture weights))
         (slots (architecture-block-slots architecture)))
    (validate-model-weights weights architecture)
    (make-model architecture
                (loop for layer below (hparam architecture 'n-layers)
                      collect (loop for slot in slots
                                    collect (make-block architecture
                                                        (architecture-block-spec architecture slot)
                                                        slot layer weights)))
                (loop for slot in (architecture-model-block-slots architecture)
                      collect (make-block architecture
                                          (architecture-model-block-spec architecture slot)
                                          slot nil weights)))))

(defgeneric layer-forward (architecture blocks x &key &allow-other-keys)
  (:documentation "Run one layer's BLOCKS on X, adding each block's delta to the residual.")
  (:method (architecture blocks x &rest keys &key &allow-other-keys)
    (declare (ignore architecture))
    (dolist (block blocks x)
      (setf x (ct:add x (apply #'block-forward block x keys))))))

(defun model-block-of-type (model type)
  (or (find-if (lambda (block) (typep block type)) (model-model-blocks model))
      (architecture-fail 'architecture-error "~S has no ~S block"
                         (class-name (class-of (model-architecture model))) type)))

(defgeneric model-forward (model tokens &key start cache all-logits))

(defmethod model-forward ((model model) tokens &key (start 0) cache all-logits)
  "Logits for TOKENS (a list or integer vector) at positions START.. Only the last position's
unless ALL-LOGITS."
  (let ((x (block-forward (model-block-of-type model 'embedding)
                          (ct:from-data (coerce tokens 'list) :dtype :s32))))
    (dolist (blocks (model-layers model))
      (setf x (layer-forward (model-architecture model) blocks x :start start :cache cache)))
    (block-forward (model-block-of-type model 'lm-head) x :last-only (not all-logits))))

(defgeneric model-ready-p (model)
  (:method (model) (declare (ignore model)) t))

(defgeneric close-model (model)
  (:method (model) (declare (ignore model)) nil))
