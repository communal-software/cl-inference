(in-package #:cl-inference/local)

(define-condition session-error (error)
  ((message :initarg :message :reader session-error-message))
  (:report (lambda (condition stream)
             (write-string (session-error-message condition) stream))))

(define-condition context-overflow (session-error) ())

(defun session-fail (control &rest args)
  (error 'session-error :message (apply #'format nil control args)))

(defun context-fail (control &rest args)
  (error 'context-overflow :message (apply #'format nil control args)))

(defstruct (kv-cache (:constructor make-kv-cache (n-ctx &key (cache-type :f32))))
  n-ctx cache-type (closed-p nil)
  (layers (make-hash-table)))

(defconstant +kv-group-size+ 32)

(defstruct (q8-buffer (:constructor %make-q8-buffer (shape bytes scales)))
  shape bytes scales)

(defun make-q8-buffer (shape)
  (destructuring-bind (capacity heads dim) shape
    (%make-q8-buffer shape
                     (make-array (* capacity heads dim) :element-type '(signed-byte 8) :initial-element 0)
                     (make-array (* capacity heads (ceiling dim +kv-group-size+))
                                 :element-type 'single-float :initial-element 1f0))))

(defun finite-kv-p (value)
  (and (not (float-features:float-nan-p value))
       (not (float-features:float-infinity-p value))))

;; TODO: scalar conversion scales with context traffic; use bulk kernels/tiled attention (#57).
(defun store-q8-rows (buffer rows start)
  (let* ((shape (q8-buffer-shape buffer)) (heads (second shape)) (dim (third shape))
         (groups (ceiling dim +kv-group-size+))
         (storage (ct:tensor-storage rows)) (strides (ct:tensor-strides rows))
         (bytes (q8-buffer-bytes buffer)) (scales (q8-buffer-scales buffer)))
    (unless (equalp (subseq (ct:tensor-shape rows) 1) (vector heads dim))
      (session-fail "Q8 cache row shape differs"))
    (dotimes (row (aref (ct:tensor-shape rows) 0))
      (dotimes (head heads)
        (dotimes (group groups)
          (let* ((first (* group +kv-group-size+)) (end (min dim (+ first +kv-group-size+)))
                 (peak 0f0) (index (+ (* (+ (* (+ start row) heads) head) dim) first))
                 (base (+ (ct:tensor-offset rows) (* row (aref strides 0)) (* head (aref strides 1)))))
            (loop for i from first below end
                  for value = (ct:storage-ref storage (+ base (* i (aref strides 2))))
                  do (unless (finite-kv-p value) (session-fail "Non-finite key or value"))
                     (setf peak (max peak (abs value))))
            (let ((scale (if (zerop peak) 1f0
                             (coerce (max (coerce least-positive-normalized-single-float 'double-float)
                                          (/ (coerce peak 'double-float) 127d0)) 'single-float))))
              (setf (aref scales (+ (* (+ (* (+ start row) heads) head) groups) group)) scale)
              (loop for i from first below end for at from index
                    for value = (ct:storage-ref storage (+ base (* i (aref strides 2))))
                    do (setf (aref bytes at) (max -127 (min 127 (round (/ value scale)))))))))))))

(defun q8-prefix (buffer end)
  (let* ((shape (q8-buffer-shape buffer)) (heads (second shape)) (dim (third shape))
         (groups (ceiling dim +kv-group-size+))
         (result (ct:zeros (list end heads dim) :dtype :f32))
         (storage (ct:tensor-storage result))
         (limit (/ most-positive-single-float 127d0))
         (bytes (q8-buffer-bytes buffer)) (scales (q8-buffer-scales buffer)))
    (dotimes (row end result)
      (dotimes (head heads)
        (dotimes (i dim)
          (let ((code (aref bytes (+ (* (+ (* row heads) head) dim) i)))
                (scale (aref scales (+ (* (+ (* row heads) head) groups) (floor i +kv-group-size+)))))
            (setf (ct:storage-ref storage (+ (* (+ (* row heads) head) dim) i))
                  (if (<= scale limit) (* code scale)
                      (coerce (max (- (coerce most-positive-single-float 'double-float))
                                   (min (coerce most-positive-single-float 'double-float)
                                        (* code (coerce scale 'double-float)))) 'single-float)))))))))

(defun cache-buffer (cache template)
  (let ((shape (cons (kv-cache-n-ctx cache) (rest (coerce (ct:tensor-shape template) 'list)))))
    (ecase (kv-cache-cache-type cache)
      (:f32 (ct:zeros shape :dtype :f32))
      (:q8 (make-q8-buffer shape)))))

;; TODO: copy-storage! moves one element at a time; a bulk copy of the contiguous rows (#39)
(defun store-rows (buffer rows start end)
  (ct:copy-storage! (ct:find-dtype (ct:tensor-dtype rows))
                    (ct:slice buffer :selectors (list (list start end)))
                    rows))

(defmethod kv-cache-update ((cache kv-cache) layer k v start)
  (when (kv-cache-closed-p cache) (session-fail "Cache is closed"))
  (let ((end (+ start (aref (ct:tensor-shape k) 0))))
    (when (> end (kv-cache-n-ctx cache))
      (context-fail "Positions up to ~D do not fit in a context of ~D" end (kv-cache-n-ctx cache)))
    (destructuring-bind (keys . values)
        (or (gethash layer (kv-cache-layers cache))
            (setf (gethash layer (kv-cache-layers cache))
                  (cons (cache-buffer cache k) (cache-buffer cache v))))
      (ecase (kv-cache-cache-type cache)
        (:f32
         (store-rows keys k start end)
         (store-rows values v start end)
         (values (ct:slice keys :selectors (list (list 0 end)))
                 (ct:slice values :selectors (list (list 0 end)))))
        (:q8
         (store-q8-rows keys k start)
         (store-q8-rows values v start)
         (values (q8-prefix keys end) (q8-prefix values end)))))))

(defgeneric make-model-cache (model n-ctx &key cache-type)
  (:method (model n-ctx &key (cache-type :f32))
    (declare (ignore model)) (make-kv-cache n-ctx :cache-type cache-type)))

(defgeneric close-cache (cache)
  (:method (cache) (declare (ignore cache)) nil))

(defmethod close-cache ((cache kv-cache))
  (clrhash (kv-cache-layers cache))
  (setf (kv-cache-closed-p cache) t)
  nil)

(defgeneric cache-storage-bytes (cache))

(defmethod cache-storage-bytes ((cache kv-cache))
  (loop for pair being the hash-values of (kv-cache-layers cache)
        sum (loop for buffer in (list (car pair) (cdr pair))
                  sum (if (q8-buffer-p buffer)
                          (+ (length (q8-buffer-bytes buffer)) (* 4 (length (q8-buffer-scales buffer))))
                          (* 4 (ct:tensor-size buffer))))))
