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

(defun record (kind name count workers function &optional prepare tokens)
  (multiple-value-bind (ms allocation) (measure function prepare)
    (format t "~&RESULT,~A,~A,~D,~D,~,4F,~D,~,4F~%" kind name count workers ms allocation
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
                                            (lambda () (local:linear x weight)))))))))))))
  (dolist (count '(1 16 128))
    (let* ((q (ct:full (list count 16 64) 0.01f0))
           (k (ct:full '(256 4 64) 0.02f0)) (v (ct:full '(256 4 64) 0.03f0))
           (mask (local::causal-mask count 256 (- 256 count) 0)))
      (with-executors
       (lambda (workers) (record "attention" "f32" count workers
                                (lambda () (local::attend q k v mask 16 4 64))))))))

(defun models ()
  (dolist (kind '(:f16 :q8_0 :qwen-f16 :qwen-q8))
    (unless (accept::verified-model-p kind) (error "Missing or changed model ~A" kind))
    (local:with-weights (weights (accept::model-path kind))
      (let* ((model (local:load-model weights)) (tokenizer (local:load-tokenizer weights))
             (base (local:encode tokenizer "The capital of France is")))
        (dolist (count '(16 128))
          (let* ((prompt (loop for i below count collect (aref base (mod i (length base)))))
                 (cache (local:make-kv-cache (+ count 8)))
                 (session (local:make-session model :n-ctx (+ count 8)))
                 (decode-token (first (tests::greedy-continuation model prompt 1))))
            (with-executors
             (lambda (workers)
               (record "prefill" kind count workers (lambda () (local:model-forward model prompt)))
               (record "decode" kind count workers
                       (lambda () (local:model-forward model (list decode-token) :start count :cache cache))
                       (lambda () (local:model-forward model prompt :cache cache)))
               (record "generate" kind count workers
                       (lambda ()
                         (multiple-value-bind (text finish ids)
                             (local:generate session prompt :max-tokens 4 :timeout 3600000)
                           (declare (ignore text))
                           (unless (and (eq finish :length) (= (length ids) 4))
                             (error "Incomplete measured generation: ~S" finish))
                           ids))
                       (lambda () (local:session-reset session)) 4)))))))))

(format t "~&PROFILE ~S ~S ~S simd=~S VECLIB_MAXIMUM_THREADS=~S~%"
        (machine-version) (lisp-implementation-type) (lisp-implementation-version)
        (trivial-simd:backend) (uiop:getenv "VECLIB_MAXIMUM_THREADS"))
(format t "kind,name,count,workers,milliseconds,lisp-bytes,tokens-per-second~%")
(synthetic)
(unless (uiop:getenv "CL_INFERENCE_PARALLEL_SYNTHETIC_ONLY") (models))
