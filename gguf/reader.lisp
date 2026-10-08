;;; Adapted from cl-qwen src/gguf.lisp
;;;
;;; MIT License
;;;
;;; Copyright (c) 2026 George Watson
;;;
;;; Permission is hereby granted, free of charge, to any person obtaining a copy
;;; of this software and associated documentation files (the "Software"), to deal
;;; in the Software without restriction, including without limitation the rights
;;; to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
;;; copies of the Software, and to permit persons to whom the Software is
;;; furnished to do so, subject to the following conditions:
;;;
;;; The above copyright notice and this permission notice shall be included in all
;;; copies or substantial portions of the Software.
;;;
;;; THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
;;; IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
;;; FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
;;; AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
;;; LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
;;; OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
;;; SOFTWARE.
(in-package #:cl-inference/gguf)

#-little-endian
(error "cl-inference/gguf reads little-endian files through native memory access")

(defconstant +magic+ #x46554747)
(defconstant +max-metadata-depth+ 32)
(defconstant +default-alignment+ 32)
(defconstant +max-alignment+ 4096)

(defstruct (ggml-type (:constructor ggml-type (id key block-size block-bytes)))
  id key block-size block-bytes)

(defparameter *ggml-types*
  (let ((table (make-hash-table)))
    (loop for (id key block-size block-bytes)
            in '((0 :f32 1 4) (1 :f16 1 2) (2 :q4_0 32 18) (3 :q4_1 32 20)
                 (6 :q5_0 32 22) (7 :q5_1 32 24) (8 :q8_0 32 34) (9 :q8_1 32 36)
                 (10 :q2_k 256 84) (11 :q3_k 256 110) (12 :q4_k 256 144)
                 (13 :q5_k 256 176) (14 :q6_k 256 210) (15 :q8_k 256 292)
                 (16 :iq2_xxs 256 66) (17 :iq2_xs 256 74) (18 :iq3_xxs 256 98)
                 (19 :iq1_s 256 50) (20 :iq4_nl 32 18) (21 :iq3_s 256 110)
                 (22 :iq2_s 256 82) (23 :iq4_xs 256 136) (24 :i8 1 1) (25 :i16 1 2)
                 (26 :i32 1 4) (27 :i64 1 8) (28 :f64 1 8) (29 :iq1_m 256 56)
                 (30 :bf16 1 2) (34 :tq1_0 256 54) (35 :tq2_0 256 66)
                 (39 :mxfp4 32 17) (40 :nvfp4 64 36) (41 :q1_0 128 18) (42 :q2_0 64 18))
          do (setf (gethash id table) (ggml-type id key block-size block-bytes)))
    table)
  "ggml type id → block geometry. Ids absent here (removed or unknown) are rejected.")

(defstruct (gguf-tensor (:include ci:tensor-info)
                        (:constructor make-gguf-tensor (name shape format-type byte-count offset)))
  (offset 0 :type (integer 0) :read-only t))

(defstruct (cursor (:constructor make-cursor (base limit)))
  base
  (limit 0 :type fixnum)
  (position 0 :type fixnum))

(defun advance (cursor count what)
  "Reserve COUNT bytes and return their start offset."
  (let ((start (cursor-position cursor)))
    (when (> count (- (cursor-limit cursor) start))
      (gguf-fail "Truncated GGUF: ~A needs ~D bytes at offset ~D" what count start))
    (setf (cursor-position cursor) (+ start count))
    start))

(defun read-scalar (cursor type what)
  (cffi:mem-ref (cursor-base cursor) type
                (advance cursor (cffi:foreign-type-size type) what)))

(defun read-count (cursor what)
  "Read a u64 count that cannot exceed the bytes left, as every item takes at least one."
  (let ((count (read-scalar cursor :uint64 what)))
    (when (> count (- (cursor-limit cursor) (cursor-position cursor)))
      (gguf-fail "~A ~D exceeds the remaining file" what count))
    count))

(defun read-gguf-string (cursor)
  (let* ((count (read-count cursor "String length"))
         (start (advance cursor count "string"))
         (base (cursor-base cursor))
         (octets (make-array count :element-type '(unsigned-byte 8))))
    (dotimes (i count)
      (setf (aref octets i) (cffi:mem-ref base :uint8 (+ start i))))
    (handler-case (babel:octets-to-string octets :encoding :utf-8 :errorp t)
      (babel:character-decoding-error () (gguf-fail "Invalid UTF-8 string")))))

(defun metadata-array-type (type)
  (case type
    (0 '(unsigned-byte 8)) (1 '(signed-byte 8))
    (2 '(unsigned-byte 16)) (3 '(signed-byte 16))
    (4 '(unsigned-byte 32)) (5 '(signed-byte 32))
    (6 'single-float) (10 '(unsigned-byte 64)) (11 '(signed-byte 64))
    (12 'double-float)
    (t t)))

(defun read-value (cursor type depth)
  (when (> depth +max-metadata-depth+)
    (gguf-fail "Metadata nesting exceeds ~D levels" +max-metadata-depth+))
  (case type
    (0 (read-scalar cursor :uint8 "u8")) (1 (read-scalar cursor :int8 "i8"))
    (2 (read-scalar cursor :uint16 "u16")) (3 (read-scalar cursor :int16 "i16"))
    (4 (read-scalar cursor :uint32 "u32")) (5 (read-scalar cursor :int32 "i32"))
    (6 (read-scalar cursor :float "f32"))
    (7 (case (read-scalar cursor :uint8 "bool")
         (0 nil) (1 t) (t (gguf-fail "Invalid boolean"))))
    (8 (read-gguf-string cursor))
    (9 (let ((element-type (read-scalar cursor :uint32 "array type")))
         (unless (<= 0 element-type 12)
           (gguf-fail "Unknown metadata type ~D" element-type))
         (let* ((count (read-count cursor "Array length"))
                (array (make-array count :element-type (metadata-array-type element-type))))
           (dotimes (i count array)
             (setf (aref array i) (read-value cursor element-type (1+ depth)))))))
    (10 (read-scalar cursor :uint64 "u64")) (11 (read-scalar cursor :int64 "i64"))
    (12 (read-scalar cursor :double "f64"))
    (t (gguf-fail "Unknown metadata type ~D" type))))

(defun read-tensor-info (cursor)
  (let* ((name (read-gguf-string cursor))
         (rank (read-scalar cursor :uint32 "tensor rank")))
    (unless (<= 1 rank 4)
      (gguf-fail "Invalid rank ~D for tensor ~A" rank name))
    (let ((dims (loop repeat rank
                      collect (let ((dim (read-scalar cursor :uint64 "tensor dimension")))
                                (unless (<= 1 dim most-positive-fixnum)
                                  (gguf-fail "Invalid dimension ~D for tensor ~A" dim name))
                                dim)))
          (type-id (read-scalar cursor :uint32 "tensor type"))
          (offset (read-scalar cursor :uint64 "tensor offset")))
      (let ((type (gethash type-id *ggml-types*)))
        (unless type
          (gguf-fail "Unsupported ggml type ~D for tensor ~A" type-id name))
        (unless (zerop (mod (first dims) (ggml-type-block-size type)))
          (gguf-fail "Tensor ~A row length ~D is not a multiple of ~D"
                     name (first dims) (ggml-type-block-size type)))
        (make-gguf-tensor name (reverse dims) (ggml-type-key type)
                          (* (/ (reduce #'* dims) (ggml-type-block-size type))
                             (ggml-type-block-bytes type))
                          offset)))))

(defun validate-alignment (alignment)
  (unless (and (integerp alignment) (<= 1 alignment +max-alignment+)
               (zerop (logand alignment (1- alignment))))
    (gguf-fail "Invalid tensor alignment ~A" alignment))
  alignment)

(defun validate-layout (tensors data-start alignment size)
  (let ((end 0))
    (dolist (tensor (sort (copy-list tensors) #'< :key #'gguf-tensor-offset))
      (let ((offset (gguf-tensor-offset tensor)))
        (unless (zerop (mod offset alignment))
          (gguf-fail "Tensor ~A offset ~D is not aligned to ~D"
                     (ci:tensor-info-name tensor) offset alignment))
        (when (< offset end)
          (gguf-fail "Tensor ~A overlaps another tensor" (ci:tensor-info-name tensor)))
        (setf end (+ offset (ci:tensor-info-byte-count tensor)))
        (when (> (+ data-start end) size)
          (gguf-fail "Tensor ~A extends past the end of the file"
                     (ci:tensor-info-name tensor)))))))

(defun parse-gguf (base size)
  "Parse the header at BASE. Return metadata table, metadata keys, tensors and data start."
  (let ((cursor (make-cursor base size)))
    (unless (= (read-scalar cursor :uint32 "magic") +magic+)
      (gguf-fail "Invalid GGUF magic"))
    (unless (= (read-scalar cursor :uint32 "version") 3)
      (gguf-fail "Only little-endian GGUF v3 is supported"))
    (let* ((tensor-count (read-count cursor "Tensor count"))
           (metadata-count (read-count cursor "Metadata count"))
           (metadata (make-hash-table :test 'equal))
           (keys '())
           (names (make-hash-table :test 'equal))
           (tensors '()))
      (dotimes (i metadata-count)
        (let ((key (read-gguf-string cursor)))
          (when (nth-value 1 (gethash key metadata))
            (gguf-fail "Duplicate metadata key ~A" key))
          (setf (gethash key metadata) (read-value cursor (read-scalar cursor :uint32 "value type") 0))
          (push key keys)))
      (dotimes (i tensor-count)
        (let ((tensor (read-tensor-info cursor)))
          (when (gethash (ci:tensor-info-name tensor) names)
            (gguf-fail "Duplicate tensor ~A" (ci:tensor-info-name tensor)))
          (setf (gethash (ci:tensor-info-name tensor) names) tensor)
          (push tensor tensors)))
      (let* ((alignment (validate-alignment (gethash "general.alignment" metadata +default-alignment+)))
             (data-start (* alignment (ceiling (cursor-position cursor) alignment)))
             (tensors (nreverse tensors)))
        (when (> data-start size)
          (gguf-fail "Missing tensor data section"))
        (validate-layout tensors data-start alignment size)
        (values metadata (nreverse keys) tensors data-start)))))
