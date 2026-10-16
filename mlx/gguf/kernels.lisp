(in-package #:cl-inference/mlx)

(cl-gpu:define-kernel-macro gguf-octet (bytes index)
  `(cl-gpu:convert cl-gpu:i32 (aref ,bytes ,index)))

(cl-gpu:define-kernel-macro gguf-half (bytes index)
  `(cl-gpu:convert cl-gpu:f32
     (cl-gpu:bitcast cl-gpu:f16
       (cl-gpu:convert cl-gpu:u16
         (logior (gguf-octet ,bytes ,index)
                 (cl-gpu:shl (gguf-octet ,bytes (+ ,index 1)) 8))))))

(cl-gpu:define-kernel-macro gguf-element (bytes base e format)
  `(cond
     ((= ,format 0)
      (* (gguf-half ,bytes ,base)
         (cl-gpu:convert cl-gpu:f32
           (- (logand (cl-gpu:shr (gguf-octet ,bytes (+ ,base 2 (mod ,e 16)))
                                 (if (< ,e 16) 0 4)) 15) 8))))
     ((= ,format 1)
      (let* ((g (/ ,e 32))
             (scale (if (< g 4) (logand (gguf-octet ,bytes (+ ,base 4 g)) 63)
                        (logior (logand (gguf-octet ,bytes (+ ,base 8 g)) 15)
                                (cl-gpu:shr (logand (gguf-octet ,bytes (+ ,base g)) 192) 2))))
             (minimum (if (< g 4) (logand (gguf-octet ,bytes (+ ,base 8 g)) 63)
                          (logior (cl-gpu:shr (gguf-octet ,bytes (+ ,base 8 g)) 4)
                                  (cl-gpu:shr (logand (gguf-octet ,bytes (+ ,base 4 g)) 192) 2))))
             (q (logand (cl-gpu:shr (gguf-octet ,bytes (+ ,base 16 (* 32 (/ g 2)) (mod ,e 32)))
                                   (* 4 (mod g 2))) 15)))
        (- (* (* (gguf-half ,bytes ,base) (cl-gpu:convert cl-gpu:f32 scale))
              (cl-gpu:convert cl-gpu:f32 q))
           (* (gguf-half ,bytes (+ ,base 2)) (cl-gpu:convert cl-gpu:f32 minimum)))))
     (t
      (let* ((h (/ ,e 128)) (g (/ (mod ,e 128) 32)) (lane (mod ,e 32))
             (low (logand (cl-gpu:shr (gguf-octet ,bytes (+ ,base (* h 64) (* (mod g 2) 32) lane))
                                     (* 4 (/ g 2))) 15))
             (high (logand (cl-gpu:shr (gguf-octet ,bytes (+ ,base 128 (* h 32) lane)) (* g 2)) 3))
             (scale (gguf-octet ,bytes (+ ,base 192 (* h 8) (* g 2) (/ lane 16)))))
        (* (* (gguf-half ,bytes (+ ,base 208))
              (cl-gpu:convert cl-gpu:f32 (if (< scale 128) scale (- scale 256))))
           (cl-gpu:convert cl-gpu:f32 (- (logior low (cl-gpu:shl high 4)) 32)))))))

(cl-gpu:define-kernel-macro gguf-dot (out x bytes columns rows format elements block-bytes)
  `(let* ((row (cl-gpu:convert cl-gpu:i32 (cl-gpu:group-id 0)))
          (batch (cl-gpu:convert cl-gpu:i32 (cl-gpu:group-id 1)))
          (lane (cl-gpu:convert cl-gpu:i32 (cl-gpu:local-id 0)))
          (sums (cl-gpu:shared-array cl-gpu:f32 32))
          (sum 0.0))
     (cl-gpu:for (column lane (< column ,columns) (incf column 32))
       (let* ((base (* (+ (* row (/ ,columns ,elements)) (/ column ,elements)) ,block-bytes))
              (value (gguf-element ,bytes base (mod column ,elements) ,format)))
         (incf sum (* (aref ,x (+ (* batch ,columns) column)) value))))
     (setf (aref sums lane) sum)
     (cl-gpu:barrier)
     (when (= lane 0)
       (let ((total 0.0))
         (dotimes (i 32) (incf total (aref sums i)))
         (setf (aref ,out (+ (* batch ,rows) row)) total)))))

(defmacro define-gguf-projection (raw wrapper)
  `(progn
     (cl-gpu:define-kernel ,raw
         ((out (cl-gpu:ptr cl-gpu:f32 :global)) (x (cl-gpu:ptr cl-gpu:f32 :global))
          (bytes (cl-gpu:ptr cl-gpu:u8 :global)) (columns cl-gpu:i32) (rows cl-gpu:i32)
          (format cl-gpu:i32) (elements cl-gpu:i32) (block-bytes cl-gpu:i32))
       (gguf-dot out x bytes columns rows format elements block-bytes))
     (cl-mlx/gpu:define-kernel ,wrapper (cl-gpu:find-kernel ',raw)
         (x bytes columns rows format elements block-bytes)
       :outputs ((out (list (first (mx:shape x)) rows)))
       :grid (list (* 32 rows) (first (mx:shape x))) :group 32
       :templates ((format format) (elements elements) (block-bytes block-bytes)))))

(define-gguf-projection gguf-decode-raw gguf-decode)
;; TODO: prefill rereads weights per token; share packed tiles across token rows (#68).
(define-gguf-projection gguf-prefill-raw gguf-prefill)

(cl-gpu:define-kernel gguf-gather-raw
    ((out (cl-gpu:ptr cl-gpu:f32 :global)) (bytes (cl-gpu:ptr cl-gpu:u8 :global))
     (tokens (cl-gpu:ptr cl-gpu:i32 :global)) (columns cl-gpu:i32) (count cl-gpu:i32)
     (format cl-gpu:i32) (elements cl-gpu:i32) (block-bytes cl-gpu:i32))
  (let ((i (cl-gpu:convert cl-gpu:i32 (cl-gpu:global-id 0))))
    (when (< i (* count columns))
      (let* ((column (mod i columns))
             (base (* (+ (* (aref tokens (/ i columns)) (/ columns elements)) (/ column elements)) block-bytes)))
        (setf (aref out i) (gguf-element bytes base (mod column elements) format))))))

(cl-mlx/gpu:define-kernel gguf-gather (cl-gpu:find-kernel 'gguf-gather-raw)
    (bytes tokens columns count format elements block-bytes)
  :outputs ((out (list count columns))) :grid (* count columns) :group 32
  :templates ((format format) (elements elements) (block-bytes block-bytes)))
