(defpackage #:cl-inference/examples/tests
  (:use #:cl #:fiveam)
  (:local-nicknames (#:examples #:cl-inference/examples) (#:fixtures #:cl-inference/local/tests)
                    (#:ct #:cl-tensor))
  (:export #:run-tests))

(in-package #:cl-inference/examples/tests)

(def-suite :cl-inference/examples)
(in-suite :cl-inference/examples)

(test local-completion-example
  (fixtures::with-gguf-file (path (fixtures::backend-fixture "<|im_start|>"))
    (let ((reply (examples:complete-file path '((:role :user :content "hi")) :max-tokens 2)))
      (is (eq :assistant (getf reply :role)))
      (is (getf reply :done))
      (is (equal '((:type :text :text "aa")) (getf reply :content))))))

(test weight-inspection-example
  (fixtures::with-gguf-file (path (fixtures::sample-gguf))
    (let ((summary (examples:inspect-weights path)))
      (is (eq :gguf (getf summary :format)))
      (is (string= "llama" (getf summary :architecture)))
      (is (find :q4_k (getf summary :tensors) :key (lambda (tensor) (getf tensor :type)))))))

(test application-architecture-example
  (let* ((spec (fixtures::spec*)) (table (fixtures::weight-table spec)) (tokens '(3 0 5)))
    (fixtures::with-gguf-file (path (fixtures::spec-gguf "observed-llama" spec table))
      (multiple-value-bind (logits observations) (examples:forward-observed path tokens)
        (is (fixtures::rows-close-p (fixtures::tensor-rows logits)
                                    (last (fixtures::ref-logits table spec :adjacent tokens))))
        (is (equal (loop for layer below (fixtures::spec-get spec :layers)
                         collect (list :layer layer :shape (list 3 (fixtures::spec-get spec :dim))))
                   observations))))))

(defun run-tests () (run! :cl-inference/examples))
