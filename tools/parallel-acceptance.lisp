(require :asdf)
(unless (find-package :ql)
  (load (merge-pathnames "quicklisp/setup.lisp" (user-homedir-pathname))))
(asdf:initialize-source-registry
 `(:source-registry (:directory ,(merge-pathnames "../" (uiop:pathname-directory-pathname *load-truename*)))
                   :inherit-configuration))
(asdf:load-system :cl-inference/local/acceptance)

(in-package #:cl-inference/acceptance)

(dolist (case *cases*)
  (unless (and (verified-model-p (case-option (first case) :model)) (oracle (first case)))
    (error "Missing acceptance input for ~A" (first case))))
(unless (run-tests) (error "Serial CPU acceptance failed"))

(let ((serial *runs*))
  (dolist (workers '(2 4))
    (multiple-value-bind (executor close) (tests::make-threaded-test-executor workers)
      (unwind-protect
           (let ((ct:*executor* executor) (*runs* (make-hash-table)))
             (format t "~&PARALLEL ACCEPTANCE workers=~D~%" workers)
             (dolist (case '(:f16 :f16-chat :q8 :q8-chat :qwen-f16 :qwen-q8 :qwen-f16-thinking
                             :qwen-q8-thinking :qwen-f16-chat :qwen-q8-chat))
               (dolist (suffix '(prompt-ids layer-traces logits greedy))
                 (let ((test (find-symbol (format nil "~A-~A" case suffix) :cl-inference/acceptance)))
                   (unless (and test (run! test)) (error "Parallel acceptance failed: ~A" test))))
               (let ((ours (our-run case)) (expected (gethash case serial)))
                 (unless (and (equal (getf ours :greedy) (getf expected :greedy))
                              (< (relative-error (getf ours :logits) (getf expected :logits)) 1d-4))
                   (error "Serial/parallel mismatch: ~A" case)))))
        (funcall close)))))
(format t "~&Serial and parallel CPU acceptance passed.~%")
