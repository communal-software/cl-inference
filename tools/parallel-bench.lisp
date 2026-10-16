(require :asdf)
(unless (find-package :ql)
  (load (merge-pathnames "quicklisp/setup.lisp" (user-homedir-pathname))))
(asdf:initialize-source-registry
 `(:source-registry (:directory ,(merge-pathnames "../" (uiop:pathname-directory-pathname *load-truename*)))
                   :inherit-configuration))
(asdf:load-system :cl-inference/local/acceptance)

(defpackage #:cl-inference/parallel-bench
  (:use #:cl)
  (:local-nicknames (#:local #:cl-inference/local) (#:ct #:cl-tensor)
                    (#:tests #:cl-inference/local/tests) (#:accept #:cl-inference/acceptance)))
(in-package #:cl-inference/parallel-bench)

(defvar *result* nil)

(defun benchmark-options (name defaults)
  (let ((value (uiop:getenv name)))
    (if value
        (mapcar (lambda (part) (intern (string-upcase part) :keyword))
                (uiop:split-string value :separator ","))
        defaults)))

(defun call-with-dispatch (path function)
  (let ((local::*half-linear-path* (if (eq path :baseline-auto) :auto path))
        (original (fdefinition 'local::half-linear-path)))
    (unwind-protect
         (progn
           (when (eq path :baseline-auto)
             (setf (fdefinition 'local::half-linear-path)
                   (lambda (x weight)
                     (let ((local::*half-linear-path* (if ct:*executor* :tiled :auto)))
                       (funcall original x weight)))))
           (funcall function))
      (setf (fdefinition 'local::half-linear-path) original))))

(defun measure (function &optional prepare)
  (let ((samples nil))
    (dotimes (i 4)
      #+sbcl (sb-ext:gc :full t)
      (when prepare (funcall prepare))
      #+sbcl (when prepare (sb-ext:gc :full t))
      (let ((start (get-internal-real-time))
            (allocation #+sbcl (sb-ext:get-bytes-consed) #-sbcl 0))
        (setf *result* (funcall function))
        (when (plusp i)
          (push (list (* 1000d0 (/ (- (get-internal-real-time) start) internal-time-units-per-second))
                      (- #+sbcl (sb-ext:get-bytes-consed) #-sbcl 0 allocation)) samples))))
    (values (second (sort (mapcar #'first samples) #'<))
            (second (sort (mapcar #'second samples) #'<)))))

(defun measure-linear (function)
  (flet ((batch (iterations)
           (let ((start (get-internal-real-time))
                 (allocation #+sbcl (sb-ext:get-bytes-consed) #-sbcl 0))
             (dotimes (i iterations) (setf *result* (funcall function)))
             (values (/ (- (get-internal-real-time) start) internal-time-units-per-second)
                     (- #+sbcl (sb-ext:get-bytes-consed) #-sbcl 0 allocation)))))
    #+sbcl (sb-ext:gc :full t)
    (funcall function)
    (let ((iterations 1))
      (loop while (< (batch iterations) 0.01d0) do (setf iterations (* iterations 2)))
      (let ((samples (loop repeat 3 collect (multiple-value-list (batch iterations)))))
        (values (* 1000d0 (/ (second (sort (mapcar #'first samples) #'<)) iterations))
                (/ (second (sort (mapcar #'second samples) #'<)) iterations))))))

(defun record (kind name count workers function &optional prepare tokens (path :auto))
  (multiple-value-bind (ms allocation)
      (if (member kind '("linear" "mapped-linear") :test #'string=)
          (measure-linear function)
          (measure function prepare))
    (format t "~&RESULT,~A,~A,~D,~D,~A,~,4F,~,3F,~,4F~%" kind name count workers path ms allocation
            (if tokens (/ (* tokens 1000d0) ms) 0d0))
    (finish-output)))

(defun with-executors (function)
  (funcall function 0)
  (dolist (workers '(2 4))
    (multiple-value-bind (executor close) (tests::make-threaded-test-executor workers)
      (unwind-protect (let ((ct:*executor* executor)) (funcall function workers)) (funcall close)))))

(defun synthetic ()
  (dolist (dtype '(:f32 :f16 :bf16 :q8_0 :q4_0 :q4_k :q6_k))
    (let* ((spec (tests::spec* :dim 256 :ffn 256 :heads 4 :kv 2 :head-dim 64 :rope-dims 64 :layers 1 :dtype dtype))
           (table (tests::weight-table spec)))
      (tests::with-spec-model (model "blk-llama" spec table)
        (let ((weight (local::attn-q (first (aref (coerce (local:model-layers model) 'vector) 0)))) )
          (dolist (count '(1 16 128))
            (let ((x (ct:full (list count 256) 0.01f0)))
              (dolist (path (if (member dtype '(:f16 :bf16)) '(:auto :encoded :tiled) '(:auto)))
                (let ((local::*half-linear-path* path))
                  (with-executors
                   (lambda (workers) (record "linear" (format nil "~A-~A" dtype path) count workers
                                            (lambda () (local:linear x weight)) nil nil path)))))))))))
  (dolist (count '(1 16 128))
    (let* ((q (ct:full (list count 16 64) 0.01f0))
           (k (ct:full '(256 4 64) 0.02f0)) (v (ct:full '(256 4 64) 0.03f0))
           (mask (local::causal-mask count 256 (- 256 count) 0)))
      (with-executors
       (lambda (workers) (record "attention" "f32" count workers
                                (lambda () (local::attend q k v mask 16 4 64))))))))

(defun mapped-projections ()
  (dolist (dtype '(:f16 :bf16))
    (uiop:with-temporary-file (:stream stream :pathname path :direction :output
                              :element-type '(unsigned-byte 8))
      (let* ((shapes '((8192 2048) (2048 8192) (3072 1024) (151936 1024)))
             (elements (reduce #'max shapes :key (lambda (shape) (* (first shape) (+ 8 (second shape))))))
             (block (make-array 65536 :element-type '(unsigned-byte 8)))
             (bits (ecase dtype (:f16 #x3800) (:bf16 #x3f00))))
        (loop for i from 0 below (length block) by 2
              do (setf (aref block i) (ldb (byte 8 0) bits)
                       (aref block (1+ i)) (ldb (byte 8 8) bits)))
        (loop for start from 0 below (* 2 elements) by (length block)
              do (write-sequence block stream :end (min (length block) (- (* 2 elements) start))))
        (finish-output stream)
        (multiple-value-bind (pointer bytes) (cl-inference/mapping:map-file path 1 #'error)
          (unwind-protect
               (let ((storage (trivial-simd:make-vector-view pointer :u16 elements)))
                 (dolist (shape (cons '(256 256) shapes))
                   (destructuring-bind (rows columns) shape
                     (dolist (layout (if (= rows 256) '(:stepped :reversed) '(:mapped :padded)))
                       (let* ((strides (case layout
                                         (:mapped (list columns 1))
                                         (:padded (list (+ columns 8) 1))
                                         (:stepped (list (* 2 columns) 2))
                                         (:reversed (list (- columns) -1))))
                              (weight (ct:make-tensor-view storage shape :dtype dtype :strides strides
                                         :offset (if (eq layout :reversed) (1- (* rows columns)) 0))))
                         (dolist (count (if (= rows 151936) '(1) '(1 2 16 128)))
                           (let ((x (ct:full (list count columns) 0.5f0)))
                             (with-executors
                              (lambda (workers)
                                (dolist (selected '(:baseline-auto :auto :encoded :tiled))
                                  (call-with-dispatch
                                   selected
                                   (lambda ()
                                     (let ((result (local:linear x weight)))
                                       (assert (eq :f32 (ct:tensor-dtype result)))
                                       (assert (= (* columns 0.25f0) (ct:tref result 0 0)
                                                  (ct:tref result (1- count) (1- rows)))))
                                     (record "mapped-linear"
                                             (format nil "~A-~Dx~D-~A" dtype rows columns layout)
                                             count workers (lambda () (local:linear x weight))
                                             nil nil selected)))))))))))))
            (cl-inference/mapping:unmap-file pointer bytes #'error)))))))

(defun models ()
  (dolist (kind (benchmark-options "CL_INFERENCE_PARALLEL_MODELS" '(:f16 :q8_0 :qwen-f16 :qwen-q8)))
    (unless (accept::verified-model-p kind) (error "Missing or changed model ~A" kind))
    (local:with-weights (weights (accept::model-path kind))
      (let* ((model (local:load-model weights)) (tokenizer (local:load-tokenizer weights))
             (base (local:encode tokenizer "The capital of France is")))
        (dolist (count '(16 128))
          (let* ((prompt (loop for i below count collect (aref base (mod i (length base)))))
                 (cache (local:make-kv-cache (+ count 8)))
                 (session (local:make-session model :n-ctx (+ count 8)))
                 (expected (tests::greedy-continuation model prompt 4)))
            (unwind-protect
                 (progn
                   (local:model-forward model prompt :cache cache)
                   (with-executors
                    (lambda (workers)
                      (dolist (path (if (member kind '(:f16 :qwen-f16))
                                       (benchmark-options "CL_INFERENCE_PARALLEL_PATHS"
                                                          '(:baseline-auto :auto :encoded :tiled))
                                       '(:auto)))
                        (call-with-dispatch
                         path
                         (lambda ()
                           (when (eq path :auto)
                             (record "prefill" kind count workers
                                     (lambda () (local:model-forward model prompt)) nil nil path))
                           (record "decode" kind count workers
                                   (lambda () (local:model-forward model (list (first expected))
                                                                   :start count :cache cache))
                                   nil nil path)
                           (when (member path '(:baseline-auto :auto))
                             (record "generate" kind count workers
                                     (lambda ()
                                       (multiple-value-bind (text finish ids)
                                           (local:generate session prompt :max-tokens 4 :timeout 3600000)
                                         (declare (ignore text))
                                         (unless (and (eq finish :length) (equal ids expected))
                                           (error "Incorrect measured generation: ~S ~S" finish ids))
                                         ids))
                                     (lambda () (local:session-reset session)) 4 path))))))))
              (local:close-session session)
              (local:close-cache cache))))))))

(format t "~&PROFILE ~S ~S ~S simd=~S VECLIB_MAXIMUM_THREADS=~S tensor=~S simd-source=~S~%"
        (machine-version) (lisp-implementation-type) (lisp-implementation-version)
        (trivial-simd:backend) (uiop:getenv "VECLIB_MAXIMUM_THREADS")
        (asdf:component-pathname (asdf:find-system :cl-tensor))
        (asdf:component-pathname (asdf:find-system :trivial-simd)))
(format t "kind,name,count,workers,path,milliseconds,lisp-bytes,tokens-per-second~%")
(cond ((uiop:getenv "CL_INFERENCE_PARALLEL_HALF_PROFILE") (mapped-projections) (models))
      ((uiop:getenv "CL_INFERENCE_PARALLEL_HALF_ONLY") (mapped-projections))
      (t
       (unless (uiop:getenv "CL_INFERENCE_PARALLEL_MODELS_ONLY") (synthetic))
       (unless (uiop:getenv "CL_INFERENCE_PARALLEL_SYNTHETIC_ONLY") (models))))
