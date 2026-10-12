(in-package #:cl-inference/gguf)

(defclass gguf-weights (local:weights)
  ((pointer :initarg :pointer :reader weights-pointer)
   (size :initarg :size :reader weights-size)
   (data-start :initarg :data-start :reader weights-data-start)
   (metadata :initarg :metadata :reader weights-metadata-table)
   (metadata-keys :initarg :metadata-keys :reader weights-keys)
   (tensors :initarg :tensors :reader weights-tensors)
   (tensor-table :reader weights-tensor-table)))

(defmethod initialize-instance :after ((weights gguf-weights) &key tensors &allow-other-keys)
  (let ((table (make-hash-table :test 'equal)))
    (dolist (tensor tensors)
      (setf (gethash (local:tensor-info-name tensor) table) tensor))
    (setf (slot-value weights 'tensor-table) table)))

(defun open-gguf (path)
  (let ((path (truename path)))
    (multiple-value-bind (pointer size) (map-file path)
      (let ((success nil))
        (unwind-protect
             (multiple-value-bind (metadata keys tensors data-start) (parse-gguf pointer size)
               (prog1 (make-instance 'gguf-weights :format :gguf :path path :pointer pointer
                                                   :size size :data-start data-start
                                                   :metadata metadata :metadata-keys keys
                                                   :tensors tensors)
                 (setf success t)))
          (unless success
            (unmap-file pointer size)))))))

(defmethod local:weights-metadata-ref ((weights gguf-weights) key)
  (gethash key (weights-metadata-table weights)))
(defmethod local:weights-metadata-key-list ((weights gguf-weights)) (weights-keys weights))
(defmethod local:weights-tensor-info-list ((weights gguf-weights)) (weights-tensors weights))
(defmethod local:weights-tensor-info-ref ((weights gguf-weights) name)
  (gethash name (weights-tensor-table weights)))
(defmethod local:weights-architecture-name ((weights gguf-weights))
  (let ((name (gethash "general.architecture" (weights-metadata-table weights))))
    (and (stringp name) name)))
(defmethod local:weights-release ((weights gguf-weights))
  (unmap-file (weights-pointer weights) (weights-size weights)))

(defun tensor-pointer (weights tensor)
  (cffi:inc-pointer (weights-pointer weights)
                    (+ (weights-data-start weights) (gguf-tensor-offset tensor))))

(defgeneric gguf-type-storage (type-key pointer element-count)
  (:documentation "Return storage and dtype for ELEMENT-COUNT elements of ggml TYPE-KEY at POINTER, or NIL.
Extension systems add methods for their types.")
  (:method (type-key pointer element-count)
    (declare (ignore type-key pointer element-count))
    nil))

(defmacro define-plain-type (type-key view-type dtype)
  `(defmethod gguf-type-storage ((type-key (eql ,type-key)) pointer element-count)
     (values (trivial-simd:make-vector-view pointer ,view-type element-count) ,dtype)))

(define-plain-type :f32 :f32 :f32)
(define-plain-type :f64 :f64 :f64)
(define-plain-type :f16 :u16 :f16)
(define-plain-type :bf16 :u16 :bf16)
(define-plain-type :i8 :s8 :s8)
(define-plain-type :i16 :s16 :s16)
(define-plain-type :i32 :s32 :s32)
(define-plain-type :i64 :s64 :s64)

(defmethod local:weights-make-tensor ((weights gguf-weights) (info gguf-tensor))
  (let ((type-key (local:tensor-info-format-type info)))
    (multiple-value-bind (storage dtype)
        (gguf-type-storage type-key (tensor-pointer weights info)
                           (reduce #'* (local:tensor-info-shape info)))
      (unless storage
        (error 'local:unsupported-weights-type
               :message (format nil "Tensor ~A has no typed view for ggml type ~A"
                                (local:tensor-info-name info) type-key)
               :tensor (local:tensor-info-name info) :format-type type-key))
      (ct:make-tensor-view storage (local:tensor-info-shape info) :dtype dtype))))

(defmethod local:weights-make-bytes ((weights gguf-weights) (info gguf-tensor))
  (let ((count (local:tensor-info-byte-count info)))
    (ct:make-tensor-view (trivial-simd:make-vector-view (tensor-pointer weights info) :u8 count)
                         (list count))))

(local:register-weights-format
 :gguf
 :detect (lambda (path header)
           (declare (ignore path))
           (and (>= (length header) 4) (equalp (subseq header 0 4) #(71 71 85 70))))
 :open #'open-gguf)
