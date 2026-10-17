(in-package #:cl-inference/hf)

(define-condition safetensors-error (local:weights-error) ())

(defun fail (control &rest args)
  (apply #'local:weights-fail 'safetensors-error control args))

(defconstant +maximum-json-bytes+ 100000000)

(defun read-json-value (parser event value depth)
  (when (> depth 128) (fail "JSON nesting exceeds 128 levels"))
  (case event
    (:value (values value nil))
    (:begin-object
     (let ((object (make-hash-table :test 'equal)) (keys nil))
       (loop
         (multiple-value-bind (event key) (json:parse-next parser)
           (when (eq event :end-object) (return (values object (nreverse keys))))
           (unless (eq event :object-key) (fail "Expected a JSON object key"))
           (when (nth-value 1 (gethash key object)) (fail "Duplicate JSON key ~S" key))
           (multiple-value-bind (event value) (json:parse-next parser)
             (setf (gethash key object) (read-json-value parser event value (1+ depth))))
           (push key keys)))))
    (:begin-array
     (let ((items nil))
       (loop
         (multiple-value-bind (event value) (json:parse-next parser)
           (when (eq event :end-array) (return (coerce (nreverse items) 'vector)))
           (push (read-json-value parser event value (1+ depth)) items)))))
    (t (fail "Incomplete JSON value"))))

(defun parse-json-object (octets)
  (let ((text (babel:octets-to-string octets :encoding :utf-8 :errorp t)))
    (json:with-parser (parser text :key-fn nil :max-string-length +maximum-json-bytes+)
      (multiple-value-bind (event value) (json:parse-next parser)
        (unless (eq event :begin-object) (fail "Expected a JSON object"))
        (multiple-value-bind (object keys) (read-json-value parser event value 0)
          (when (json:parse-next parser) (fail "Trailing JSON content"))
          (values object keys))))))

(defun read-json-file (path)
  (with-open-file (stream path :element-type '(unsigned-byte 8))
    (let ((size (file-length stream)))
      (when (> size +maximum-json-bytes+) (fail "JSON file ~A exceeds 100 MB" path))
      (let ((octets (make-array size :element-type '(unsigned-byte 8))))
        (unless (= size (read-sequence octets stream)) (fail "Truncated JSON file ~A" path))
        (parse-json-object octets)))))
