(in-package #:cl-inference/local/tests)

;;; Test-only GGUF writer. Knobs produce the malformed files the reader must reject.

(defun make-buffer ()
  (make-array 0 :element-type '(unsigned-byte 8) :adjustable t :fill-pointer 0))

(defun put-uint (buffer value bytes)
  (dotimes (i bytes)
    (vector-push-extend (ldb (byte 8 (* 8 i)) value) buffer)))

(defun put-octets (buffer octets)
  (map nil (lambda (octet) (vector-push-extend octet buffer)) octets))

(defun put-string (buffer string)
  "STRING is a Lisp string or a raw octet vector."
  (let ((octets (if (stringp string) (babel:string-to-octets string :encoding :utf-8) string)))
    (put-uint buffer (length octets) 8)
    (put-octets buffer octets)))

(defun float-bits (value type)
  (cffi:with-foreign-object (pointer type)
    (setf (cffi:mem-ref pointer type) value)
    (cffi:mem-ref pointer (if (eq type :float) :uint32 :uint64))))

(defun put-value (buffer type value)
  "TYPE is a GGUF value type id; arrays are (:array element-type items)."
  (ecase type
    (0 (put-uint buffer value 1)) (1 (put-uint buffer (ldb (byte 8 0) value) 1))
    (2 (put-uint buffer value 2)) (3 (put-uint buffer (ldb (byte 16 0) value) 2))
    (4 (put-uint buffer value 4)) (5 (put-uint buffer (ldb (byte 32 0) value) 4))
    (6 (put-uint buffer (float-bits value :float) 4))
    (7 (put-uint buffer (if (integerp value) value (if value 1 0)) 1))
    (8 (put-string buffer value))
    (9 (destructuring-bind (element-type items) (rest value)
         (put-uint buffer element-type 4)
         (put-uint buffer (length items) 8)
         (dolist (item items) (put-value buffer element-type item))))
    (10 (put-uint buffer value 8)) (11 (put-uint buffer (ldb (byte 64 0) value) 8))
    (12 (put-uint buffer (float-bits value :double) 8))
    (99 (put-uint buffer value 4))))

(defun nested-array (depth)
  "An array of arrays DEPTH levels deep, ending in an empty array."
  (if (zerop depth)
      '(:array 0 ())
      `(:array 9 (,(nested-array (1- depth))))))

(defun pad-to (buffer alignment)
  (loop until (zerop (mod (length buffer) alignment)) do (vector-push-extend 0 buffer)))

(defun build-gguf (&key metadata tensors (alignment 32) (version 3) (magic #x46554747)
                     tensor-count truncate (pad t))
  "METADATA is a list of (key type value). TENSORS is a list of
(name dims type-id octets &optional offset); offsets default to the next aligned slot.
TRUNCATE drops that many bytes from the end. PAD nil leaves the header unaligned, as
llama.cpp writes a file with no tensors."
  (let ((header (make-buffer)) (data (make-buffer)) (cursor 0)
        (metadata (if (= alignment 32)
                      metadata
                      (append metadata `(("general.alignment" 4 ,alignment))))))
    (put-uint header magic 4)
    (put-uint header version 4)
    (put-uint header (or tensor-count (length tensors)) 8)
    (put-uint header (length metadata) 8)
    (loop for (key type value) in metadata
          do (put-string header key)
             (put-uint header type 4)
             (put-value header type value))
    (loop for (name dims type-id octets explicit-offset) in tensors
          for offset = (or explicit-offset cursor)
          do (put-string header name)
             (put-uint header (length dims) 4)
             (dolist (dim dims) (put-uint header dim 8))
             (put-uint header type-id 4)
             (put-uint header offset 8)
             (loop while (< (length data) offset) do (vector-push-extend 0 data))
             (loop for i from 0 for octet across octets
                   do (if (< (+ offset i) (length data))
                          (setf (aref data (+ offset i)) octet)
                          (vector-push-extend octet data)))
             (setf cursor (* alignment (ceiling (length data) alignment))))
    (when pad (pad-to header alignment))
    (put-octets header data)
    (let ((out (coerce header '(simple-array (unsigned-byte 8) (*)))))
      (if truncate (subseq out 0 (- (length out) truncate)) out))))

(defun f32-octets (values)
  (let ((buffer (make-buffer)))
    (dolist (value values buffer) (put-uint buffer (float-bits value :float) 4))))

(defun bits-float (bits)
  (cffi:with-foreign-object (pointer :uint32)
    (setf (cffi:mem-ref pointer :uint32) bits)
    (cffi:mem-ref pointer :float)))

(defun f16-bits (value)
  "IEEE half-precision bits of VALUE, rounded to nearest even."
  (let ((sign (if (minusp (float-sign value)) #x8000 0))
        (x (abs (rational value))))
    (if (zerop x)
        sign
        (let ((e (- (integer-length (numerator x)) (integer-length (denominator x)))))
          (when (< x (expt 2 e)) (decf e))
          (let* ((e (max e -14))
                 (mantissa (round x (expt 2 (- e 10)))))
            (when (>= mantissa 2048) (setf mantissa (floor mantissa 2)) (incf e))
            (if (and (= e -14) (< mantissa 1024))
                (logior sign mantissa)
                (logior sign (ash (+ e 15) 10) (- mantissa 1024))))))))

(defun f16-value (bits)
  (let ((exponent (ldb (byte 5 10) bits)) (fraction (ldb (byte 10 0) bits)))
    (* (if (logbitp 15 bits) -1 1)
       (if (zerop exponent)
           (float (* fraction (expt 2 -24)) 1f0)
           (float (* (+ 1024 fraction) (expt 2 (- exponent 25))) 1f0)))))

(defun bf16-bits (value)
  "The top half of VALUE's single-float bits, rounded to nearest even."
  (let ((bits (float-bits value :float)))
    (ldb (byte 16 0) (ash (+ bits #x7fff (ldb (byte 1 16) bits)) -16))))

(defun bf16-value (bits) (bits-float (ash bits 16)))

(defun u16-octets (values)
  (let ((buffer (make-buffer)))
    (dolist (value values buffer) (put-uint buffer value 2))))

(defun write-octets (path octets)
  (with-open-file (stream path :direction :output :if-exists :supersede
                              :element-type '(unsigned-byte 8))
    (write-sequence octets stream)))

(defun read-octets (path)
  (with-open-file (stream path :element-type '(unsigned-byte 8))
    (let ((octets (make-array (file-length stream) :element-type '(unsigned-byte 8))))
      (read-sequence octets stream)
      octets)))

(defmacro with-gguf-file ((path octets) &body body)
  `(uiop:with-temporary-file (:pathname ,path :stream stream :element-type '(unsigned-byte 8))
     (write-sequence ,octets stream)
     :close-stream
     ,@body))
