(in-package #:cl-inference/local/tests)

(in-suite :cl-inference/local)

(defun q8-spec (&rest overrides)
  "A model whose every matrix row is a whole number of Q8_0 blocks."
  (apply #'spec* :dtype :q8_0 :dim 32 :heads 4 :kv 2 :head-dim 8 :rope-dims 8 :ffn 64 :vocab 9
         overrides))

(defun q8-weight (rows columns)
  "A (ROWS COLUMNS) Q8_0 tensor, and its dequantized values by row."
  (let* ((values (weight-values (format nil "w~D.~D" rows columns) (list rows columns)))
         (octets (coerce (q8-0-octets values) '(simple-array (unsigned-byte 8) (*)))))
    (values (ct:make-tensor-view (cl-inference/quant:make-q8-0-storage octets) (list rows columns)
                                 :dtype :q8_0)
            (loop for r below rows
                  collect (subseq (round-through :q8_0 values) (* r columns) (* (1+ r) columns))))))

(defun q8-activations (rows columns)
  (ct:from-data (loop for r below rows
                      collect (loop for c below columns
                                    collect (* 0.1f0 (- (mod (+ (* r 5) (* c 3)) 11) 5))))))

(test q8-0-model-matches-reference
  (check-model "blk-llama" :adjacent (q8-spec))
  (check-model "blk-llama" :adjacent (q8-spec :tied t))
  (check-model "blk-qwen" :neox (q8-spec :qk-norm t :freqs t)))

(test q8-0-cache-matches-a-full-pass
  (let* ((spec (q8-spec)) (table (weight-table spec)) (tokens '(3 0 5 2 6)))
    (with-spec-model (model "blk-llama" spec table)
      (let ((cache (make-instance 'test-cache)))
        (local:model-forward model (subseq tokens 0 3) :cache cache)
        (is (rows-close-p (tensor-rows (local:model-forward model (subseq tokens 3) :start 3
                                                                                    :cache cache))
                          (last (ref-logits table spec :adjacent tokens))))))))

(test q8-0-matmul-matches-dequantized-values
  (multiple-value-bind (weight rows) (q8-weight 40 96)
    (dolist (count '(1 3 6 9))
      (let ((x (q8-activations count 96)))
        (is (rows-close-p (tensor-rows (ct:matmul x (ct:transpose weight)))
                          (loop for r below count
                                collect (loop for row in rows
                                              collect (ref-dot (mapcar (lambda (v) (coerce v 'double-float))
                                                                       (loop for c below 96 collect (ct:tref x r c)))
                                                               (mapcar (lambda (v) (coerce v 'double-float)) row))))
                          1d-3)
            "~D activation rows" count)))))

(test q8-0-matmul-accepts-an-offset-activation-view
  (multiple-value-bind (weight) (q8-weight 5 64)
    (let* ((x (q8-activations 4 64)) (view (ct:slice x :selectors (list (list 1 3)))))
      (is (rows-close-p (tensor-rows (ct:matmul view (ct:transpose weight)))
                        (last (tensor-rows (ct:matmul (ct:slice x :selectors (list (list 0 3)))
                                                      (ct:transpose weight)))
                              2)
                        1d-5)))))

(test q8-0-take-and-dequantize-agree
  (multiple-value-bind (weight rows) (q8-weight 6 64)
    (let ((taken (ct:take weight (ct:from-data '(4 1 4) :dtype :s32))))
      (is (eq :q8_0 (ct:tensor-dtype taken)))
      (is (equalp #(3 64) (ct:tensor-shape taken)))
      (is (rows-close-p (tensor-rows (ct:astype taken :f32))
                        (list (nth 4 rows) (nth 1 rows) (nth 4 rows)) 1d-6))
      (is (rows-close-p (tensor-rows (ct:dequantize weight)) rows 1d-6)))))

(test q8-0-dequantizes-a-transposed-view
  (multiple-value-bind (weight rows) (q8-weight 3 32)
    (let ((flipped (ct:astype (ct:transpose weight) :f32)))
      (is (equalp #(32 3) (ct:tensor-shape flipped)))
      (dotimes (r 3)
        (dotimes (c 32)
          (is (= (nth c (nth r rows)) (ct:tref flipped c r))))))))

(test q8-0-views-are-whole-blocks
  (multiple-value-bind (weight) (q8-weight 4 64)
    (let ((storage (ct:tensor-storage weight)))
      (is (= 256 (ct:storage-length storage)))
      (signals error (ct:make-tensor-view storage (list 4 16) :dtype :q8_0))
      (signals error (ct:make-tensor-view storage (list 3 64) :dtype :q8_0 :offset 16))
      (signals error (ct:make-tensor-view storage (list 64 4) :dtype :q8_0))
      (is (ct:make-tensor-view storage (list 2 64) :dtype :q8_0 :offset 128))
      (signals error (ct:slice weight :selectors (list :all (list 0 16)))))
    (signals error (cl-inference/quant:make-q8-0-storage
                    (make-array 35 :element-type '(unsigned-byte 8))))))

(test q8-0-matmul-needs-the-weight-on-the-right
  (multiple-value-bind (weight) (q8-weight 4 32)
    (signals error (ct:matmul (ct:transpose weight) (q8-activations 4 32)))
    (signals error (ct:matmul (q8-activations 2 32) weight))))

(test q8-0-gguf-tensors-read-the-file-in-place
  (let ((values (weight-values "q8" '(2 32))))
    (with-gguf-file (path (build-gguf :metadata '(("general.architecture" 8 "llama"))
                                      :tensors `(("q8" (32 2) 8 ,(q8-0-octets values)))))
      (local:with-weights (weights path :format :gguf)
        (let ((tensor (local:weights-tensor weights "q8")))
          (is (eq :q8_0 (ct:tensor-dtype tensor)))
          (is (equalp #(2 32) (ct:tensor-shape tensor)))
          (is (typep (cl-inference/quant:q8-0-storage-bytes (ct:tensor-storage tensor))
                     'trivial-simd:vector-view))
          (is (rows-close-p (tensor-rows (ct:astype tensor :f32))
                            (list (subseq (round-through :q8_0 values) 0 32)
                                  (subseq (round-through :q8_0 values) 32))
                            1d-6)))))))
