(in-package #:cl-inference/acceptance)

(def-suite :cl-inference/quant/acceptance :in :cl-inference/acceptance)
(in-suite :cl-inference/quant/acceptance)

(defparameter *quant-manifest*
  (com.inuoe.jzon:parse (asdf:system-relative-pathname "cl-inference" "tests/acceptance/quant-models.json")))

(defun register-quant-model (family dtype)
  (let* ((key (format nil "~(~A~)-~(~A~)" family dtype))
         (entry (gethash key *quant-manifest*)) (model (intern (string-upcase key) :keyword)))
    (unless entry (error "Missing pinned quantized model ~A" key))
    (push (list model (gethash "file" entry) (gethash "sha256" entry) :dtype dtype) *models*)
    (dolist (mode (if (eq family :qwen) '(:plain :chat :thinking) '(:plain :chat)))
      (let ((case (intern (string-upcase (format nil "~A-~A" key mode)) :keyword)))
        (push (append (list case :model model :layers (if (eq family :llama) 16 28)
                           :quant t :tolerances '(0.9999d0 0.02d0 5))
                      (if (eq mode :plain) (list :text "The capital of France is")
                          (list :date "26 Jul 2024" :thinking (eq mode :thinking)
                                :messages '((:role :system :content "Answer in one word.")
                                            (:role :user :content "What is the capital of France?")))))
              *cases*)))))

(dolist (family '(:llama :qwen))
  (dolist (dtype '(:q4_0 :q4_k :q6_k)) (register-quant-model family dtype)))

(defmacro define-quant-tests ()
  `(progn
     ,@(loop for family in '(llama qwen) append
         (loop for dtype in '(q4_0 q4_k q6_k) append
           (loop for mode in (if (eq family 'qwen) '(plain chat thinking) '(plain chat))
                 collect `(define-case-tests ,(intern (format nil "~A-~A-~A" family dtype mode) :keyword)))))))

(define-quant-tests)

(test quant-model-tensor-coverage
  (dolist (family '(:llama :qwen))
    (dolist (dtype '(:q4_0 :q4_k :q6_k))
      (let* ((key (intern (format nil "~A-~A" family dtype) :keyword))
             (case (intern (format nil "~A-PLAIN" key) :keyword)))
        (with-acceptance (case (ours oracle))
          (declare (ignore ours oracle))
          (local:with-weights (weights (model-path key))
            (let ((types (remove-duplicates
                          (mapcar (lambda (name) (local:tensor-info-format-type (local:weights-tensor-info weights name)))
                                  (local:weights-tensor-names weights)))))
              (is (member dtype types))
              (when (eq dtype :q4_k) (is (member :q6_k types)))
              (is (subsetp types '(:f32 :f16 :bf16 :q8_0 :q4_0 :q4_k :q6_k))))))))))
