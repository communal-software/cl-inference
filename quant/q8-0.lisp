(in-package #:cl-inference/quant)

;;; Q8_0: blocks of 32 elements, each an f16 scale then 32 signed bytes, as GGUF stores them.

(defconstant +block-elements+ 32)
(defconstant +block-bytes+ 34)

(defclass q8-0 (ct:dtype) ())

(defclass q8-0-storage ()
  ((bytes :initarg :bytes :reader q8-0-storage-bytes)))

(defun byte-count (bytes)
  (if (simd:vector-view-p bytes) (simd:vector-view-length bytes) (length bytes)))

(defun make-q8-0-storage (bytes)
  "Storage over BYTES, an (unsigned-byte 8) vector or view of whole blocks."
  (unless (zerop (mod (byte-count bytes) +block-bytes+))
    (error "Q8_0 data of ~D bytes is not a whole number of blocks" (byte-count bytes)))
  (make-instance 'q8-0-storage :bytes bytes))

(defmethod ct:storage-length ((storage q8-0-storage))
  (* +block-elements+ (floor (byte-count (q8-0-storage-bytes storage)) +block-bytes+)))
(defmethod ct:storage-element-type ((storage q8-0-storage)) 'single-float)
(defmethod ct:storage-dtype ((storage q8-0-storage)) :q8_0)

(defmethod ct:allocate-storage ((dtype q8-0) shape)
  (let ((size (reduce #'* shape)))
    (unless (zerop (mod size +block-elements+))
      (error "Q8_0 storage needs a multiple of ~D elements, not ~D" +block-elements+ size))
    (make-q8-0-storage (make-array (* (floor size +block-elements+) +block-bytes+)
                                   :element-type '(unsigned-byte 8) :initial-element 0))))

(defun row-major-p (shape strides)
  (equalp strides (ct:row-major-strides shape)))

(defun transposed-matrix-p (shape strides)
  "Whether SHAPE and STRIDES are the transpose of a row-major matrix."
  (and (= 2 (length shape)) (= 1 (aref strides 0)) (= (aref shape 0) (aref strides 1))))

(defmethod ct:validate-storage-view ((dtype q8-0) storage shape strides offset)
  (unless (typep storage 'q8-0-storage)
    (error "Q8_0 tensors need Q8_0 storage"))
  (unless (and (zerop (mod offset +block-elements+))
               (plusp (length shape))
               (or (and (row-major-p shape strides)
                        (zerop (mod (aref shape (1- (length shape))) +block-elements+)))
                   (and (transposed-matrix-p shape strides)
                        (zerop (mod (aref shape 0) +block-elements+)))))
    (error "Q8_0 views are whole rows of ~D-element blocks, or their transpose" +block-elements+)))

(defmethod ct:copy-storage-supported-p ((dtype q8-0) out)
  (typep (ct:tensor-storage out) 'q8-0-storage))

(defmethod ct:copy-storage! ((dtype q8-0) out input)
  (unless (and (eq (ct:tensor-dtype out) (ct:tensor-dtype input))
               (equalp (ct:tensor-shape out) (ct:tensor-shape input))
               (ct:contiguous-p out) (ct:contiguous-p input))
    (error "Q8_0 copies need contiguous tensors of one shape"))
  (flet ((byte-start (tensor) (* +block-bytes+ (floor (ct:tensor-offset tensor) +block-elements+))))
    (let ((count (* +block-bytes+ (floor (ct:tensor-size out) +block-elements+))))
      (simd:copy! (q8-0-storage-bytes (ct:tensor-storage out))
                  (q8-0-storage-bytes (ct:tensor-storage input))
                  :start 0 :end count
                  :destination-start (byte-start out) :source-start (byte-start input))))
  out)

;;; Matmul

(defun matmul-weight-p (weight columns)
  "Whether WEIGHT is a row-major (m COLUMNS) Q8_0 matrix viewed as its (COLUMNS m) transpose."
  (and (eq (ct:tensor-dtype weight) :q8_0)
       (zerop (ct:tensor-offset weight))
       (transposed-matrix-p (ct:tensor-shape weight) (ct:tensor-strides weight))
       (= columns (aref (ct:tensor-shape weight) 0))))

(defun matmul-executor (out inputs options)
  ;; TODO: the kernel costs about a millisecond per activation row; many rows are faster as dequantized tiles through BLAS (#41)
  (declare (ignore options))
  (destructuring-bind (x weight) inputs
    (let ((x (if (and (ct:contiguous-p x) (zerop (ct:tensor-offset x))) x (ct:copy-tensor x))))
      (simd:q8-0-matmul! (ct:tensor-storage out) (ct:tensor-storage x)
                         (q8-0-storage-bytes (ct:tensor-storage weight))
                         (aref (ct:tensor-shape x) 0) (aref (ct:tensor-shape weight) 1)
                         (aref (ct:tensor-shape x) 1)))))

;;; Dequantizing

(defun block-scale (low high)
  (let* ((bits (logior low (ash high 8)))
         (sign (if (logbitp 15 bits) -1f0 1f0))
         (exponent (ldb (byte 5 10) bits))
         (fraction (ldb (byte 10 0) bits)))
    (cond ((zerop exponent) (* sign (scale-float (float fraction 1f0) -24)))
          ((= exponent 31) (error "Q8_0 scale is not finite"))
          (t (* sign (scale-float (float (+ 1024 fraction) 1f0) (- exponent 25)))))))

(defun dequantize-executor (out inputs options)
  (declare (ignore options))
  (let* ((input (first inputs)) (shape (ct:tensor-shape input)) (strides (ct:tensor-strides input))
         (first-block (floor (ct:tensor-offset input) +block-elements+))
         (reach (1+ (loop for size across shape for stride across strides
                          sum (* (1- size) stride))))
         (block-count (ceiling (+ (mod (ct:tensor-offset input) +block-elements+) reach)
                               +block-elements+))
         (bytes (make-array (* block-count +block-bytes+) :element-type '(unsigned-byte 8)))
         (target (ct:tensor-storage out))
         (position 0))
    (when (zerop (ct:tensor-size input)) (return-from dequantize-executor))
    (simd:copy! bytes (q8-0-storage-bytes (ct:tensor-storage input))
                :start 0 :end (length bytes) :source-start (* first-block +block-bytes+))
    (labels ((element (index)
               (multiple-value-bind (block within) (floor (- index (* first-block +block-elements+))
                                                          +block-elements+)
                 (let* ((base (* block +block-bytes+)) (byte (aref bytes (+ base 2 within))))
                   (* (block-scale (aref bytes base) (aref bytes (1+ base)))
                      (if (< byte 128) byte (- byte 256))))))
             (walk (axis index)
               (if (= axis (length shape))
                   (progn (setf (aref target position) (element index)) (incf position))
                   (dotimes (i (aref shape axis))
                     (walk (1+ axis) (+ index (* i (aref strides axis))))))))
      (walk 0 (ct:tensor-offset input)))))

(defmethod ct:resolve-operation ((dtype q8-0) operation inputs options)
  (case operation
    (:matmul
     (destructuring-bind (&optional x weight) inputs
       (when (and (ct:tensorp x) (ct:tensorp weight)
                  (eq (ct:tensor-dtype x) :f32) (= 2 (length (ct:tensor-shape x)))
                  (matmul-weight-p weight (aref (ct:tensor-shape x) 1)))
         (values :f32 #'matmul-executor))))
    ((:convert :dequantize)
     (when (and (eq (ct:tensor-dtype (first inputs)) :q8_0) (eq (getf options :dtype) :f32))
       (values :f32 #'dequantize-executor)))))

(ct:register-dtype
 (or (ignore-errors (ct:find-dtype :q8_0))
     (make-instance 'q8-0 :name :q8_0 :element-type 'single-float
                          :block-size +block-elements+ :block-bytes +block-bytes+
                          :storage-only t)))

;;; GGUF

(defmethod gguf:gguf-type-storage ((type-key (eql :q8_0)) pointer element-count)
  (values (make-q8-0-storage
           (simd:make-vector-view pointer :u8 (* (floor element-count +block-elements+) +block-bytes+)))
          :q8_0))
