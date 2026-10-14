(in-package #:cl-inference/local)

(defclass local-backend (ci:backend)
  ((weights :initarg :weights :initform nil :reader backend-weights)
   (model :initarg :model :reader backend-model)
   (tokenizer :initarg :tokenizer :reader backend-tokenizer)
   (template :initarg :template :reader backend-template)
   (n-ctx :initarg :n-ctx :reader backend-n-ctx)
   (n-batch :initarg :n-batch :reader backend-n-batch)
   (policy :initarg :session-policy :initform :serialized :reader backend-session-policy)
   (session :initform nil :accessor backend-session)
   (lock :initform (bt:make-lock) :reader backend-session-lock)))

(defun backend-new-session (backend)
  (make-session (backend-model backend) :tokenizer (backend-tokenizer backend)
                :chat-template (backend-template backend) :n-ctx (backend-n-ctx backend)
                :n-batch (backend-n-batch backend)))

(defun make-local-backend (weights &key name n-ctx (n-batch 512) (session-policy :serialized))
  "A backend borrowing open WEIGHTS. SESSION-POLICY is :SERIALIZED (one reusable cache) or :PER-CALL."
  (unless (member session-policy '(:serialized :per-call))
    (session-fail "session-policy must be :serialized or :per-call"))
  (let* ((model (load-model weights))
         (backend (make-instance 'local-backend :name name :weights weights :model model
                                 :tokenizer (load-tokenizer weights)
                                 :template (load-chat-template weights)
                                 :n-ctx (or n-ctx (default-n-ctx model)) :n-batch n-batch
                                 :session-policy session-policy)))
    (let ((session (backend-new-session backend)))
      (when (eq session-policy :serialized) (setf (backend-session backend) session)))
    backend))

(defmethod ci:describe-backend ((backend local-backend))
  (let ((weights (backend-weights backend)))
    (list :kind :provider :name (ci:backend-name backend) :summary "In-process model"
          :status (if (and weights (not (weights-closed-p weights))) :ready :unavailable)
          :session-policy (backend-session-policy backend)
          :params '((:temperature number) (:top-p number) (:top-k (integer 1))
                    (:seed integer) (:max-tokens (integer 0)) (:stop (ci:array-of string))))))

(defvar *active-local-backends* nil)

(defun local-request-check (request)
  (let ((timeout (getf request :timeout ci:+default-timeout+))
        (max (getf request :max-tokens)) (stop (getf request :stop))
        (cancel (getf request :cancel)))
    (cond
      ((not (and (realp timeout) (>= timeout 0))) "timeout must be non-negative milliseconds")
      ((not (or (null max) (typep max '(integer 0)))) "max-tokens must be non-negative")
      ((not (and (listp stop) (every (lambda (s) (and (stringp s) (plusp (length s)))) stop)))
       "stop must be a list of non-empty strings")
      ((not (or (null cancel) (ci:cancel-token-p cancel))) "cancel must be a cancel token")
      ((not (listp (getf request :tools))) "tools must be a list")
      ((some (lambda (message)
               (some (lambda (block) (not (eq (getf block :type) :text)))
                     (ci:normalize-content (getf message :content))))
             (getf request :messages))
       "Unsupported: local models accept text content only")
      (t (dolist (tool (getf request :tools))
           (unless (and (listp tool) (evenp (length tool)) (keywordp (getf tool :name)))
             (session-fail "A tool needs a keyword name"))
           (ci:validate-schema (ci:tool-schema tool)))
         nil))))

(defun local-generate (backend session request)
  (let* ((parser-problem nil)
         (tools (getf request :tools))
         (parser (make-reply-parser (backend-template backend) :tools tools
                                    :sink (getf request :stream) :ref (getf request :ref)))
         (sampler (handler-case
                      (make-sampler :temperature (getf request :temperature 1)
                                    :top-p (getf request :top-p 1) :top-k (getf request :top-k)
                                    :seed (getf request :seed))
                    (sampler-error (e) (session-fail "~A" e))))
         (tokens (encode-chat (backend-template backend) (backend-tokenizer backend)
                              (getf request :messages) :tools tools)))
    (session-reset session)
    (multiple-value-bind (raw finish ids usage)
        (generate session tokens :special t
                  :sampler (lambda (logits)
                             (when parser-problem (error parser-problem))
                             (sample sampler logits))
                  :max-tokens (getf request :max-tokens)
                  :stop (getf request :stop) :timeout (getf request :timeout ci:+default-timeout+)
                  :cancel (getf request :cancel)
                  ;; GENERATE ignores sink errors; defer parser failures to the decoding loop.
                  :sink (lambda (event)
                          (when (eq (getf event :type) :text-delta)
                            (unless parser-problem
                              (handler-case (reply-parser-push parser (getf event :text))
                                (error (e) (setf parser-problem e)))))))
      (declare (ignore raw ids))
      (when parser-problem (error parser-problem))
      (if (ci:result-error-p finish)
          finish
          (multiple-value-bind (text calls reasoning) (reply-parser-finish parser)
            (ci::make-reply text calls (if (and calls (eq finish :stop)) :tool-calls finish)
                            (append (list :usage usage)
                                    (when reasoning (list :reasoning reasoning)))))))))

(defun local-session-completion (backend request)
  (if (eq (backend-session-policy backend) :per-call)
      (local-generate backend (backend-new-session backend) request)
      (if (member backend *active-local-backends*)
          (ci:bad-request "Reentrant completion needs :session-policy :per-call")
          (bt:with-lock-held ((backend-session-lock backend))
            (let ((*active-local-backends* (cons backend *active-local-backends*))
                  (success nil))
              (unwind-protect
                   (let ((result (local-generate backend
                                                 (or (backend-session backend)
                                                     (setf (backend-session backend)
                                                           (backend-new-session backend)))
                                                 request)))
                     (setf success (not (ci:result-error-p result)))
                     result)
                (unless success (setf (backend-session backend) nil))))))))

(defmethod ci:backend-complete ((backend local-backend) request)
  (let ((problem (handler-case (local-request-check request)
                   (error (e) (princ-to-string e)))))
    (when problem
      (return-from ci:backend-complete
        (ci:call-with-completion-deadline
         (list :stream (getf request :stream) :ref (getf request :ref))
         (lambda () (ci:bad-request "~A" problem))))))
  (ci:call-with-completion-deadline
   request
   (lambda ()
     (handler-case
         (if (and (backend-weights backend) (not (weights-closed-p (backend-weights backend))))
             (local-session-completion backend request)
             (ci:fail :unavailable))
       (reply-parser-error (e) (ci:backend-error :invalid-output (princ-to-string e)))
       (sampler-error (e) (ci:backend-error :inference (princ-to-string e)))
       (session-error (e) (ci:bad-request "~A" e))
       (chat-template-error (e) (ci:bad-request "~A" e))
       (error (e) (ci:backend-error :inference (princ-to-string e)))))))

(ci:register-backend (make-instance 'local-backend :name :local))
