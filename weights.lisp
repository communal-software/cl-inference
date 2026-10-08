(in-package #:cl-inference)

(define-condition weights-error (error)
  ((message :initarg :message :reader weights-error-message))
  (:report (lambda (condition stream)
             (write-string (weights-error-message condition) stream))))

(define-condition unknown-weights-format (weights-error)
  ((format :initarg :format :reader unknown-weights-format-key)))

(define-condition weights-closed (weights-error) ())

(define-condition unsupported-weights-type (weights-error)
  ((tensor :initarg :tensor :reader unsupported-weights-type-tensor)
   (format-type :initarg :format-type :reader unsupported-weights-type-format-type)))

(defun weights-fail (condition-type control &rest args)
  (error condition-type :message (apply #'format nil control args)))

(defstruct (tensor-info (:constructor make-tensor-info (name shape format-type byte-count))
                        (:copier nil))
  (name "" :type string :read-only t)
  (shape nil :type list :read-only t)
  (format-type nil :read-only t)
  (byte-count 0 :type (integer 0) :read-only t))

(defclass weights ()
  ((format :initarg :format :reader weights-format)
   (path :initarg :path :reader weights-path)
   (closed-p :initform nil :accessor weights-closed-p)))

(defgeneric weights-metadata-ref (weights key)
  (:documentation "Return the metadata value for KEY and whether it is present."))
(defgeneric weights-metadata-key-list (weights))
(defgeneric weights-tensor-info-list (weights)
  (:documentation "Return every tensor-info in file order."))
(defgeneric weights-tensor-info-ref (weights name)
  (:documentation "Return the tensor-info called NAME, or NIL."))
(defgeneric weights-make-tensor (weights info)
  (:documentation "Return a typed tensor view, or signal unsupported-weights-type."))
(defgeneric weights-make-bytes (weights info)
  (:documentation "Return a :u8 tensor view of the raw payload."))
(defgeneric weights-release (weights)
  (:documentation "Free format resources; called once by close-weights."))

(defvar *weights-formats* nil
  "Alist of (key detect open) in registration order.")

(defun register-weights-format (key &key detect open)
  "DETECT is (lambda (path header-octets)) → true for a file of this format.
OPEN is (lambda (path)) → a weights instance."
  (let ((entry (list key detect open))
        (existing (assoc key *weights-formats*)))
    (if existing
        (setf (cdr existing) (cdr entry))
        (setf *weights-formats* (append *weights-formats* (list entry))))
    key))

(defun read-header-octets (path)
  (with-open-file (stream path :element-type '(unsigned-byte 8))
    (let* ((buffer (make-array 16 :element-type '(unsigned-byte 8)))
           (count (read-sequence buffer stream)))
      (subseq buffer 0 count))))

(defun find-weights-format (key)
  (or (assoc key *weights-formats*)
      (weights-fail 'unknown-weights-format "Unknown weights format ~S" key)))

(defun detect-weights-format (path)
  (let ((header (read-header-octets path)))
    (or (find-if (lambda (entry) (funcall (second entry) path header)) *weights-formats*)
        (weights-fail 'unknown-weights-format "No weights format recognises ~A" path))))

(defun open-weights (path &key format)
  "Open the weights file at PATH. FORMAT names a registered format; NIL detects it."
  (let ((entry (if format (find-weights-format format) (detect-weights-format path))))
    (funcall (third entry) path)))

(defun close-weights (weights)
  "Release WEIGHTS. Tensor views taken from it are invalid afterwards."
  (unless (weights-closed-p weights)
    (setf (weights-closed-p weights) t)
    (weights-release weights))
  (values))

(defmacro with-weights ((var path &rest open-args) &body body)
  `(let ((,var (open-weights ,path ,@open-args)))
     (unwind-protect (progn ,@body)
       (close-weights ,var))))

(defun ensure-open (weights)
  (when (weights-closed-p weights)
    (weights-fail 'weights-closed "Weights ~A are closed" (weights-path weights)))
  weights)

(defun weights-metadata (weights key &optional default)
  "Return the value for KEY, or DEFAULT, and whether KEY was present."
  (multiple-value-bind (value present-p) (weights-metadata-ref (ensure-open weights) key)
    (if present-p
        (values value t)
        (values default nil))))

(defun weights-metadata-keys (weights)
  (weights-metadata-key-list (ensure-open weights)))

(defun weights-tensor-names (weights)
  (mapcar #'tensor-info-name (weights-tensor-info-list (ensure-open weights))))

(defun weights-tensor-info (weights name)
  (or (weights-tensor-info-ref (ensure-open weights) name)
      (weights-fail 'weights-error "No tensor ~S in ~A" name (weights-path weights))))

(defun weights-tensor (weights name)
  "Return tensor NAME as a zero-copy cl-tensor view."
  (weights-make-tensor weights (weights-tensor-info weights name)))

(defun weights-tensor-bytes (weights name)
  "Return the raw payload of tensor NAME as a zero-copy :u8 view."
  (weights-make-bytes weights (weights-tensor-info weights name)))
