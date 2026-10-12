(in-package #:cl-inference/local/tests)

(in-suite :cl-inference/local)

(defclass mock-weights (local:weights)
  ((metadata :initform (let ((table (make-hash-table :test 'equal)))
                         (setf (gethash "general.name" table) "mock"
                               (gethash "flag" table) nil)
                         table)
             :reader mock-metadata)
   (data :initform (make-array 6 :element-type 'single-float
                                 :initial-contents '(0f0 1f0 2f0 3f0 4f0 5f0))
         :reader mock-data)
   (released :initform nil :accessor mock-released)))

(defparameter *mock-info*
  (local:make-tensor-info "w" '(2 3) :f32 24))

(defmethod local:weights-metadata-ref ((w mock-weights) key)
  (gethash key (mock-metadata w)))
(defmethod local:weights-metadata-key-list ((w mock-weights))
  (sort (loop for key being the hash-keys of (mock-metadata w) collect key) #'string<))
(defmethod local:weights-tensor-info-list ((w mock-weights)) (list *mock-info*))
(defmethod local:weights-tensor-info-ref ((w mock-weights) name)
  (when (string= name "w") *mock-info*))
(defmethod local:weights-make-tensor ((w mock-weights) info)
  (ct:make-tensor-view (mock-data w) (local:tensor-info-shape info)))
(defmethod local:weights-make-bytes ((w mock-weights) info)
  (declare (ignore info))
  (ct:make-tensor-view (make-array 24 :element-type '(unsigned-byte 8)) '(24)))
(defmethod local:weights-release ((w mock-weights))
  (setf (mock-released w) t))

(local:register-weights-format
 :mock
 :detect (lambda (path header)
           (declare (ignore path))
           (and (>= (length header) 4)
                (equalp (subseq header 0 4) #(77 79 67 75))))
 :open (lambda (path) (make-instance 'mock-weights :format :mock :path path)))

(defmacro with-mock-file ((path contents) &body body)
  `(uiop:with-temporary-file (:pathname ,path :stream stream :element-type '(unsigned-byte 8))
     (write-sequence ,contents stream)
     :close-stream
     ,@body))

(defparameter *mock-magic* (coerce #(77 79 67 75 0 0 0 0) '(simple-array (unsigned-byte 8) (*))))
(defparameter *other-magic* (coerce #(1 2 3 4 5 6 7 8) '(simple-array (unsigned-byte 8) (*))))

(test weights-detects-format
  (with-mock-file (path *mock-magic*)
    (local:with-weights (w path)
      (is (eq :mock (local:weights-format w))))))

(test weights-explicit-format-skips-detection
  (with-mock-file (path *other-magic*)
    (local:with-weights (w path :format :mock)
      (is (eq :mock (local:weights-format w))))))

(test weights-unknown-format
  (with-mock-file (path *other-magic*)
    (signals local:unknown-weights-format (local:open-weights path))
    (signals local:unknown-weights-format (local:open-weights path :format :nope))))

(test weights-metadata-default-and-presence
  (with-mock-file (path *mock-magic*)
    (local:with-weights (w path)
      (is (equal '("mock" t) (multiple-value-list (local:weights-metadata w "general.name"))))
      (is (equal '(nil t) (multiple-value-list (local:weights-metadata w "flag" :x))))
      (is (equal '(:x nil) (multiple-value-list (local:weights-metadata w "absent" :x))))
      (is (equal '("flag" "general.name") (local:weights-metadata-keys w))))))

(test weights-tensor-access
  (with-mock-file (path *mock-magic*)
    (local:with-weights (w path)
      (is (equal '("w") (local:weights-tensor-names w)))
      (is (equal '(2 3) (local:tensor-info-shape (local:weights-tensor-info w "w"))))
      (is (= 4 (ct:tref (local:weights-tensor w "w") 1 1)))
      (is (equal '(24) (coerce (ct:tensor-shape (local:weights-tensor-bytes w "w")) 'list)))
      (signals local:weights-error (local:weights-tensor w "missing")))))

(test weights-close-is-idempotent-and-guards-access
  (with-mock-file (path *mock-magic*)
    (let ((w (local:open-weights path)))
      (local:close-weights w)
      (local:close-weights w)
      (is (mock-released w))
      (signals local:weights-closed (local:weights-metadata w "general.name"))
      (signals local:weights-closed (local:weights-tensor-names w))
      (signals local:weights-closed (local:weights-tensor w "w")))))

(test with-weights-closes-on-unwind
  (with-mock-file (path *mock-magic*)
    (let (captured)
      (ignore-errors
       (local:with-weights (w path)
         (setf captured w)
         (error "boom")))
      (is (local:weights-closed-p captured)))))
