(in-package #:cl-inference/local/tests)

(in-suite :cl-inference/local)

(defun packed-product-close-p (actual x weight)
  (loop for i below (aref (ct:tensor-shape x) 0) always
    (loop for j below (aref (ct:tensor-shape weight) 0) always
      (let ((sum 0d0) (magnitude 0d0))
        (dotimes (e (aref (ct:tensor-shape x) 1))
          (let ((product (* (coerce (ct:tref x i e) 'double-float) (ct:tref weight j e))))
            (incf sum product) (incf magnitude (abs product))))
        (<= (abs (- (ct:tref actual i j) sum)) (+ 1d-5 (* 1d-5 magnitude)))))))

(test packed-dequantization-matches-llama-cpp
  (dolist (fixture *packed-golden-blocks*)
    (destructuring-bind (dtype bytes expected) fixture
      (let* ((elements (ct:dtype-block-size dtype))
             (weight (ct:make-tensor-view (packed-storage-for dtype bytes) (list 5 elements) :dtype dtype))
             (decoded (ct:dequantize weight))
             (transposed (ct:astype (ct:transpose weight) :f32)))
        (dotimes (r 5)
          (dotimes (c elements)
            (is (= (aref expected (+ (* r elements) c)) (ct:tref decoded r c)))
            (is (= (ct:tref decoded r c) (ct:tref transposed c r)))))
        (let ((taken (ct:take weight (ct:from-data '(4 1 4) :dtype :s32))))
          (is (eq dtype (ct:tensor-dtype taken)))
          (is (rows-close-p (tensor-rows (ct:dequantize taken))
                            (mapcar (lambda (r) (coerce (subseq expected (* r elements) (* (1+ r) elements)) 'list))
                                    '(4 1 4)) 1d-7)))
        (is (rows-close-p (tensor-rows (ct:dequantize (ct:copy-tensor weight)))
                          (tensor-rows decoded) 1d-7))
        (signals ct:unsupported-operation (ct:add weight weight))
        (signals ct:unsupported-operation (ct:astype weight :f64))))))

(test packed-views-and-matmul
  (dolist (fixture *packed-golden-blocks*)
    (destructuring-bind (dtype bytes expected) fixture
      (declare (ignore expected))
      (let* ((k (ct:dtype-block-size dtype)) (storage (packed-storage-for dtype bytes))
             (weight (ct:make-tensor-view storage (list 5 k) :dtype dtype)))
        (signals error (ct:make-tensor-view storage (list 5 (1- k)) :dtype dtype))
        (signals error (ct:make-tensor-view storage (list 4 k) :dtype dtype :offset 1))
        (signals error (ct:slice weight :selectors (list :all (list 0 16))))
        (signals error (packed-storage-for dtype (subseq bytes 0 (1- (length bytes)))))
        (signals error (packed-storage-for dtype #(1 2 3)))
        (signals error (packed-storage-for dtype (make-array (length bytes) :element-type '(unsigned-byte 8)
                                                            :adjustable t)))
        (let ((bad (copy-seq bytes))
              (offset (ecase dtype (:q4_0 1) (:q4_k 3) (:q6_k 209))))
          (setf (aref bad offset) 124)
          (signals error
            (ct:dequantize (ct:make-tensor-view (packed-storage-for dtype bad) (list 5 k) :dtype dtype))))
        (is (equalp #(0 5) (ct:tensor-shape (ct:matmul (ct:zeros (list 0 k)) (ct:transpose weight)))))
        (dolist (n '(1 3 5 9))
          (let ((x (q8-activations n k)))
            (is (packed-product-close-p (ct:matmul x (ct:transpose weight)) x (ct:dequantize weight))
                "~S n=~D" dtype n)))
        (let* ((x (ct:slice (q8-activations 4 k) :selectors (list (list 1 3))))
               (view (ct:make-tensor-view storage (list 3 k) :dtype dtype :offset k)))
          (is (packed-product-close-p (ct:matmul x (ct:transpose view)) x (ct:dequantize view))))
        (signals ct:unsupported-operation (ct:matmul (ct:transpose weight) (q8-activations 5 k)))))))

(test packed-gguf-uses-mapped-storage
  (dolist (fixture *packed-golden-blocks*)
    (destructuring-bind (dtype bytes expected) fixture
      (let ((elements (ct:dtype-block-size dtype)))
        (with-gguf-file (path (build-gguf :tensors `(("packed" (,elements 5) ,(dtype-id dtype) ,bytes))))
          (local:with-weights (weights path)
            (let ((tensor (local:weights-tensor weights "packed")))
              (is (eq dtype (ct:tensor-dtype tensor)))
              (is (trivial-simd:vector-view-p
                   (cl-inference/quant::packed-bytes (ct:tensor-storage tensor))))
              (is (equalp expected (ct:tensor-storage (ct:dequantize (ct:reshape tensor (list (* elements 5))))))))))))))

(test packed-models-and-cache-match-reference
  (dolist (dtype '(:q4_0 :q4_k :q6_k))
    (check-model "blk-llama" :adjacent (packed-spec dtype))
    (check-model "blk-llama" :adjacent (packed-spec dtype :tied t))
    (check-model "blk-qwen" :neox (packed-spec dtype :qk-norm t :freqs t))
    (let* ((spec (packed-spec dtype :tensor-dtypes '(("blk.0.attn_v.weight" . :q6_k))))
           (table (weight-table spec)) (tokens '(3 0 5)))
      (with-spec-model (model "blk-llama" spec table)
        (let ((cache (make-instance 'test-cache)))
          (local:model-forward model '(3 0) :cache cache)
          (is (rows-close-p (tensor-rows (local:model-forward model '(5) :start 2 :cache cache))
                            (last (ref-logits table spec :adjacent tokens)))))))))
