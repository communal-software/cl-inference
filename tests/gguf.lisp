(in-package #:cl-inference/tests)

(in-suite :cl-inference)

(defparameter *q8-block*
  (let ((octets (make-buffer)))
    (put-uint octets #x3C00 2)
    (dotimes (i 32 octets) (put-uint octets (ldb (byte 8 0) (- i 16)) 1))))

(defparameter *q4k-block*
  (make-array 144 :element-type '(unsigned-byte 8) :initial-element 7))

(defun sample-gguf (&rest overrides)
  (apply #'build-gguf
         (append overrides
                 (list :metadata '(("general.architecture" 8 "llama"))
                       :tensors `(("f32" (3 2) 0 ,(f32-octets '(0f0 1f0 2f0 3f0 4f0 5f0)))
                                  ("f16" (2) 1 ,(u16-octets '(#x3C00 #x4000)))
                                  ("bf16" (2) 30 ,(u16-octets '(#x3F80 #x4000)))
                                  ("q8" (32) 8 ,*q8-block*)
                                  ("q4k" (256) 12 ,*q4k-block*))))))

(defmacro with-sample ((weights &rest overrides) &body body)
  `(with-gguf-file (path (sample-gguf ,@overrides))
     (ci:with-weights (,weights path :format :gguf)
       ,@body)))

(defun open-bad (octets)
  (with-gguf-file (path octets)
    (ci:open-weights path :format :gguf)))

(defun tensor-list (tensor)
  (loop for i below (first (coerce (ct:tensor-shape tensor) 'list))
        collect (ct:tref tensor i)))

(test gguf-detected-by-magic
  (with-gguf-file (path (sample-gguf))
    (ci:with-weights (w path)
      (is (eq :gguf (ci:weights-format w)))
      (is (typep w 'gguf:gguf-weights)))))

(test gguf-metadata-value-types
  (let ((octets (build-gguf
                 :metadata `(("u8" 0 200) ("i8" 1 -3) ("u16" 2 60000) ("i16" 3 -300)
                             ("u32" 4 4000000000) ("i32" 5 -5) ("f32" 6 1.5f0)
                             ("bool" 7 t) ("str" 8 "héllo") ("u64" 10 18446744073709551615)
                             ("i64" 11 -7) ("f64" 12 2.25d0)
                             ("u32s" 9 (:array 4 (1 2 3)))
                             ("f32s" 9 (:array 6 (0.5f0 -1.5f0)))
                             ("strs" 9 (:array 8 ("a" "bc")))
                             ("nested" 9 (:array 9 ((:array 4 (1)) (:array 4 (2 3)))))))))
    (with-gguf-file (path octets)
      (ci:with-weights (w path)
        (flet ((m (key) (ci:weights-metadata w key)))
          (is (= 200 (m "u8"))) (is (= -3 (m "i8"))) (is (= 60000 (m "u16")))
          (is (= -300 (m "i16"))) (is (= 4000000000 (m "u32"))) (is (= -5 (m "i32")))
          (is (= 1.5f0 (m "f32"))) (is (eq t (m "bool"))) (is (string= "héllo" (m "str")))
          (is (= 18446744073709551615 (m "u64"))) (is (= -7 (m "i64")))
          (is (= 2.25d0 (m "f64")))
          (is (equalp #(1 2 3) (m "u32s")))
          (is (typep (m "u32s") '(simple-array (unsigned-byte 32) (*))))
          (is (typep (m "f32s") '(simple-array single-float (*))))
          (is (equalp #("a" "bc") (m "strs")))
          (is (equalp #(#(1) #(2 3)) (m "nested"))))
        (is (equal '("u8" "i8") (subseq (ci:weights-metadata-keys w) 0 2)))))))

(test gguf-tensor-listing-and-shape
  (with-sample (w)
    (is (equal '("f32" "f16" "bf16" "q8" "q4k") (ci:weights-tensor-names w)))
    (let ((info (ci:weights-tensor-info w "f32")))
      (is (equal '(2 3) (ci:tensor-info-shape info)))
      (is (eq :f32 (ci:tensor-info-format-type info)))
      (is (= 24 (ci:tensor-info-byte-count info))))
    (is (= 34 (ci:tensor-info-byte-count (ci:weights-tensor-info w "q8"))))
    (is (= 144 (ci:tensor-info-byte-count (ci:weights-tensor-info w "q4k"))))))

(test gguf-f32-view-is-row-major
  (with-sample (w)
    (let ((tensor (ci:weights-tensor w "f32")))
      (is (eq :f32 (ct:tensor-dtype tensor)))
      (is (= 3 (ct:tref tensor 1 0)))
      (is (= 5 (ct:tref tensor 1 2))))))

(test gguf-half-precision-views
  (with-sample (w)
    (let ((f16 (ct:astype (ci:weights-tensor w "f16") :f32))
          (bf16 (ct:astype (ci:weights-tensor w "bf16") :f32)))
      (is (equal '(1f0 2f0) (tensor-list f16)))
      (is (equal '(1f0 2f0) (tensor-list bf16))))))

(test gguf-q8-0-exposes-raw-bytes-only
  (with-sample (w)
    (signals ci:unsupported-weights-type (ci:weights-tensor w "q8"))
    (signals ci:unsupported-weights-type (ci:weights-tensor w "q4k"))
    (let ((bytes (ci:weights-tensor-bytes w "q8")))
      (is (eq :u8 (ct:tensor-dtype bytes)))
      (is (equalp (coerce *q8-block* 'vector) (coerce (tensor-list bytes) 'vector))))))

(test gguf-custom-dtype-hook
  (with-sample (w)
    (let ((method (defmethod gguf:gguf-type-storage ((type (eql :q8_0)) pointer count)
                    (values (trivial-simd:make-vector-view pointer :u8 (* 34 (/ count 32))) :u8))))
      (unwind-protect
           (is (eq :u8 (ct:tensor-dtype (ci:weights-tensor w "q8"))))
        (remove-method #'gguf:gguf-type-storage method)))))

(test gguf-writes-are-copy-on-write
  (with-gguf-file (path (sample-gguf))
    (let ((before (read-octets path)))
      (ci:with-weights (w path :format :gguf)
        (let ((tensor (ci:weights-tensor w "f32")))
          (setf (ct:tref tensor 0 0) 9f0)
          (is (= 9 (ct:tref tensor 0 0)))))
      (is (equalp before (read-octets path))))))

(test gguf-close-guards-access
  (with-gguf-file (path (sample-gguf))
    (let ((w (ci:open-weights path)))
      (ci:close-weights w)
      (ci:close-weights w)
      (signals ci:weights-closed (ci:weights-tensor w "f32")))))

(test gguf-custom-alignment
  (with-gguf-file (path (sample-gguf :alignment 64))
    (ci:with-weights (w path :format :gguf)
      (is (= 5 (ct:tref (ci:weights-tensor w "f32") 1 2)))
      (is (equalp (coerce *q8-block* 'vector)
                  (coerce (tensor-list (ci:weights-tensor-bytes w "q8")) 'vector))))))

(test gguf-rejects-empty-and-short-files
  (signals gguf:gguf-error (open-bad (make-array 0 :element-type '(unsigned-byte 8))))
  (signals gguf:gguf-error (open-bad (subseq (sample-gguf) 0 10))))

(test gguf-rejects-bad-header
  (signals gguf:gguf-error (open-bad (sample-gguf :magic #x12345678)))
  (signals gguf:gguf-error (open-bad (sample-gguf :version 2)))
  (signals gguf:gguf-error (open-bad (sample-gguf :version #x03000000))))

(test gguf-rejects-truncation
  (signals gguf:gguf-error (open-bad (sample-gguf :truncate 1)))
  (signals gguf:gguf-error (open-bad (subseq (sample-gguf) 0 60)))
  (signals gguf:gguf-error (open-bad (sample-gguf :tensor-count 1000000))))

(test gguf-rejects-duplicates
  (signals gguf:gguf-error
    (open-bad (build-gguf :metadata '(("k" 4 1) ("k" 4 2)))))
  (signals gguf:gguf-error
    (open-bad (build-gguf :tensors `(("t" (2) 0 ,(f32-octets '(1f0 2f0)))
                                     ("t" (2) 0 ,(f32-octets '(1f0 2f0))))))))

(test gguf-rejects-excess-nesting
  (with-gguf-file (path (build-gguf :metadata `(("n" 9 ,(nested-array 30)))))
    (ci:with-weights (w path :format :gguf)
      (is (ci:weights-metadata w "n"))))
  (signals gguf:gguf-error (open-bad (build-gguf :metadata `(("n" 9 ,(nested-array 40)))))))

(test gguf-rejects-invalid-values
  (signals gguf:gguf-error (open-bad (build-gguf :metadata '(("b" 7 2)))))
  (signals gguf:gguf-error (open-bad (build-gguf :metadata '(("u" 99 1)))))
  (signals gguf:gguf-error
    (open-bad (build-gguf :metadata `(("s" 8 ,(coerce #(255 254) '(vector (unsigned-byte 8)))))))))

(test gguf-rejects-invalid-alignment
  (signals gguf:gguf-error (open-bad (sample-gguf :alignment 48)))
  (signals gguf:gguf-error (open-bad (sample-gguf :alignment 8192))))

(test gguf-rejects-bad-tensors
  (flet ((one (dims type &optional offset)
           (build-gguf :tensors `(("t" ,dims ,type ,(f32-octets '(1f0 2f0 3f0 4f0)) ,@(when offset (list offset)))))))
    (signals gguf:gguf-error (open-bad (one '() 0)))
    (signals gguf:gguf-error (open-bad (one '(1 1 1 1 1) 0)))
    (signals gguf:gguf-error (open-bad (one '(0) 0)))
    (signals gguf:gguf-error (open-bad (one '(4) 99)))
    (signals gguf:gguf-error (open-bad (one '(4) 4)))
    (signals gguf:gguf-error (open-bad (one '(33) 8)))
    (signals gguf:gguf-error (open-bad (one '(4) 0 1)))))

(test gguf-rejects-overlapping-tensors
  (signals gguf:gguf-error
    (open-bad (build-gguf :tensors `(("a" (4) 0 ,(f32-octets '(1f0 2f0 3f0 4f0)) 0)
                                     ("b" (4) 0 ,(f32-octets '(1f0 2f0 3f0 4f0)) 0))))))
