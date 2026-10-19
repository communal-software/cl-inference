(in-package #:cl-inference/mlx)

(defstruct gguf-packed-weight bytes shape format elements block-bytes)

(defclass mlx-gguf-weights (mlx-weights)
  ((packed :initform (make-hash-table :test 'equal) :reader packed-weights)))

(defmethod prepare-mlx-weights ((source cl-inference/gguf:gguf-weights))
  (make-instance 'mlx-gguf-weights :source source :format :gguf :path (local:weights-path source)))

(defmethod validate-mlx-source ((weights mlx-gguf-weights) architecture)
  ;; Container checks belong to the GGUF reader; shared block bindings check model shapes.
  (when (typep architecture 'local:gemma3)
    (local::validate-model-weights (source-weights weights) architecture)))

(defun gguf-block-layout (type)
  (ecase type (:q4_0 (values 0 32 18)) (:q4_k (values 1 256 144)) (:q6_k (values 2 256 210)) (:q8_0 (values 3 32 34))))

(defun validate-gguf-scales (pointer count type block-bytes name)
  (loop for base below count by block-bytes do
    (dolist (offset (ecase type (:q4_0 '(1)) (:q4_k '(1 3)) (:q6_k '(209)) (:q8_0 '(1))))
      (when (= (logand (cffi:mem-aref pointer :uint8 (+ base offset)) #x7c) #x7c)
        (model-fail "Non-finite ~A scale in ~A at byte ~D" type name base)))))

(defmethod local:weights-make-tensor ((weights mlx-gguf-weights) info)
  (let* ((name (local:tensor-info-name info)) (type (local:tensor-info-format-type info))
         (shape (local:tensor-info-shape info)))
    (cond
      ((member type '(:f32 :f16 :bf16)) (call-next-method))
      ((member type '(:q4_0 :q4_k :q6_k :q8_0))
       (or (gethash name (packed-weights weights))
           (multiple-value-bind (format elements block-bytes) (gguf-block-layout type)
             (unless (and (= (length shape) 2) (every #'plusp shape)
                          (zerop (mod (second shape) elements))
                          (= (local:tensor-info-byte-count info)
                             (* (first shape) (/ (second shape) elements) block-bytes))
                          (<= (reduce #'* shape) 2147483647)
                          (<= (* 32 (first shape)) 2147483647))
               (model-fail "Invalid ~A packed matrix dimensions/bytes for ~A: ~S" type name shape))
             (let* ((source (source-weights weights))
                    (bytes (local:weights-tensor-bytes source name))
                    (pointer (trivial-simd:vector-view-pointer (ct:tensor-storage bytes))))
               (validate-gguf-scales pointer (local:tensor-info-byte-count info) type block-bytes name)
               (let ((array (mx:from-pointer pointer (list (local:tensor-info-byte-count info)) :dtype :u8)))
                 (setf (gethash name (imported-arrays weights)) array
                       (gethash name (packed-weights weights))
                       (make-gguf-packed-weight :bytes array :shape shape :format format
                                                :elements elements :block-bytes block-bytes)))))))
      (t (error 'local:unsupported-weights-type :tensor name :format-type type
                :message (format nil "MLX GGUF does not support ~A tensor ~A" type name))))))

(defmethod weight-shape ((weight gguf-packed-weight)) (gguf-packed-weight-shape weight))

(defmethod mlx-linear (x (weight gguf-packed-weight))
  (let ((shape (mx:shape x)) (columns (second (weight-shape weight))))
    (unless (and (= (length shape) 2) (= (second shape) columns)
                 (eq (mx:dtype x) :f32))
      (model-fail "Packed projection requires an F32 matrix with ~D columns" columns))
    (first (funcall (if (= (first shape) 1) #'gguf-decode #'gguf-prefill)
                    x (gguf-packed-weight-bytes weight) columns (first (weight-shape weight))
                    (gguf-packed-weight-format weight) (gguf-packed-weight-elements weight)
                    (gguf-packed-weight-block-bytes weight)))))

(defmethod mlx-embedding ((weight gguf-packed-weight) tokens)
  (unless (and (= (length (mx:shape tokens)) 1) (eq (mx:dtype tokens) :s32))
    (model-fail "Packed embeddings require an S32 token vector"))
  (first (gguf-gather (gguf-packed-weight-bytes weight) tokens (second (weight-shape weight))
                       (mx:size tokens) (gguf-packed-weight-format weight)
                       (gguf-packed-weight-elements weight) (gguf-packed-weight-block-bytes weight))))
