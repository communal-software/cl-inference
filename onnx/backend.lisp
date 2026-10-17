(in-package #:cl-inference/onnx)

(defclass onnx-backend (local:local-backend) ())

(defun make-onnx-backend (directory &key graph name n-ctx (n-batch 512))
  "Own a CPU decoder session loaded from DIRECTORY and an explicit relative GRAPH filename."
  (let ((model (load-onnx-model directory graph)) (success nil))
    (unwind-protect
         (let* ((files (onnx-model-files model))
                (context (or n-ctx (min 4096 (onnx-model-context model))))
                (tokenizer (local:load-tokenizer files))
                (template (local:load-chat-template files)))
           (unless (and (typep context '(integer 1)) (<= context (onnx-model-context model))
                        (typep n-batch '(integer 1)))
             (fail "Invalid context or prefill batch size"))
           (unless (= (local:tokenizer-vocab-size tokenizer) (onnx-model-vocab model))
             (fail "Tokenizer vocabulary and ONNX logits vocabulary disagree"))
           (unless (typep template 'hf::smollm2-chat)
             (fail "ONNX v1 requires the SmolLM2 text chat template"))
           (prog1 (make-instance 'onnx-backend :name name :weights files :model model
                                 :tokenizer tokenizer :template template :n-ctx context :n-batch n-batch
                                 :session-policy :serialized :cache-type :f32 :reuse-prefix nil)
             (setf success t)))
      (unless success (local:close-model model)))))

(defmethod ci:describe-backend ((backend onnx-backend))
  (let ((description (call-next-method)))
    (setf (getf description :summary) "Local ONNX Runtime CPU decoder"
          (getf description :execution-provider) :cpu
          (getf description :runtime-version) (ox:runtime-version))
    description))

(defmethod ci:backend-complete ((backend onnx-backend) request)
  (ci:call-with-completion-deadline
   request
   (lambda ()
     (handler-case
         (let ((problem (local::local-request-check request)))
           (cond (problem (ci:bad-request "~A" problem))
                 ((or (getf request :tools)
                      (some (lambda (message) (or (eq (getf message :role) :tool) (getf message :tool-calls)))
                            (getf request :messages)))
                  (ci:bad-request "ONNX v1 supports text chat without tools"))
                 ((member backend local::*active-local-backends*)
                  (ci:bad-request "Reentrant ONNX completion is unsupported"))
                 (t
                  (bt:with-lock-held ((local::backend-session-lock backend))
                    (if (local:model-ready-p (local::backend-model backend))
                        (let ((local::*active-local-backends* (cons backend local::*active-local-backends*))
                              (session nil))
                          (unwind-protect
                               (progn
                                 (let ((ci::*defer-exchange-interrupt* t))
                                   (setf session (local::backend-new-session backend)))
                                 (ci::check-exchange-reason ci::*exchange*)
                                 (local::local-generate backend session request))
                            (let ((ci::*defer-exchange-interrupt* t))
                              (when session (local:close-session session)))))
                        (ci:fail :unavailable))))))
       (local:session-error (e) (ci:bad-request "~A" e))
       (local:sampler-error (e) (ci:bad-request "~A" e))
       (local:chat-template-error (e) (ci:bad-request "~A" e))
       (error (e) (ci:backend-error :inference (princ-to-string e)))))))

(defun close-onnx-backend (backend)
  (when (member backend local::*active-local-backends*)
    (fail "Cannot close an ONNX backend from its own stream callback"))
  (local:close-local-backend backend))

(defmacro with-onnx-backend ((backend directory &rest options) &body body)
  `(let ((,backend (make-onnx-backend ,directory ,@options)))
     (unwind-protect (progn ,@body) (close-onnx-backend ,backend))))
