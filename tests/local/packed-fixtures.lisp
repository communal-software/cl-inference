(in-package #:cl-inference/local/tests)

(defun pack-test-values (dtype values)
  "Encode synthetic blocks with uniform subscales; return bytes and independently rounded values."
  (let* ((elements (ct:dtype-block-size dtype)) (size (ct:dtype-block-bytes dtype))
         (limit (if (eq dtype :q6_k) 31 7)) (buffer (make-buffer)) (rounded nil))
    (loop for start from 0 below (length values) by elements do
      (let* ((chunk (subseq values start (+ start elements)))
             (bits (f16-bits (/ (reduce #'max chunk :key #'abs) limit)))
             (scale (f16-value bits))
             (quants (mapcar (lambda (v) (if (zerop scale) 0 (max (- limit) (min limit (round v scale))))) chunk))
             (bytes (make-array size :element-type '(unsigned-byte 8) :initial-element 0)))
        (flet ((half (offset) (setf (aref bytes offset) (ldb (byte 8 0) bits)
                                  (aref bytes (1+ offset)) (ldb (byte 8 8) bits))))
          (ecase dtype
            (:q4_0
             (half 0)
             (dotimes (j 16)
               (setf (aref bytes (+ 2 j)) (logior (+ 8 (nth j quants)) (ash (+ 8 (nth (+ 16 j) quants)) 4)))))
            (:q4_k
             (half 0) (half 2)
             (dotimes (j 4) (setf (aref bytes (+ 4 j)) 1 (aref bytes (+ 8 j)) 8
                                  (aref bytes (+ 12 j)) #x81))
             (dotimes (group 4)
               (dotimes (j 32)
                 (setf (aref bytes (+ 16 (* group 32) j))
                       (logior (+ 8 (nth (+ (* group 64) j) quants))
                               (ash (+ 8 (nth (+ (* group 64) 32 j) quants)) 4))))))
            (:q6_k
             (half 208)
             (fill bytes 1 :start 192 :end 208)
             (dotimes (half 2)
               (dotimes (j 32)
                 (let ((q (loop for group below 4 collect (+ 32 (nth (+ (* half 128) (* group 32) j) quants)))))
                   (setf (aref bytes (+ (* half 64) j))
                         (logior (logand 15 (first q)) (ash (logand 15 (third q)) 4))
                         (aref bytes (+ (* half 64) 32 j))
                         (logior (logand 15 (second q)) (ash (logand 15 (fourth q)) 4))
                         (aref bytes (+ 128 (* half 32) j))
                         (loop for value in q for group from 0 sum (ash (ash value -4) (* group 2)))))))))
          (put-octets buffer bytes)
          (dolist (q quants) (push (* scale q) rounded)))))
    (values buffer (nreverse rounded))))

(defun packed-storage-for (dtype bytes)
  (funcall (ecase dtype
             (:q4_0 #'cl-inference/quant:make-q4-0-storage)
             (:q4_k #'cl-inference/quant:make-q4-k-storage)
             (:q6_k #'cl-inference/quant:make-q6-k-storage)) bytes))

(defun packed-spec (dtype &rest overrides)
  (apply #'spec* :dtype dtype :dim 256 :heads 2 :kv 1 :head-dim 128 :rope-dims 128
                 :ffn 256 :layers 1 :vocab 7 overrides))
