(in-package #:cl-inference/client)

;;; A backend answers one neutral request with one result. HTTP protocols,
;;; providers layered over them and in-process engines all implement
;;; BACKEND-COMPLETE; callers name a backend by keyword or hold the object.

(defclass backend ()
  ((name :initarg :name :initform nil :reader backend-name))
  (:documentation "Anything COMPLETE can address."))

(defgeneric backend-complete (backend request)
  (:documentation "Perform one turn against BACKEND for REQUEST, a plist already
checked against the contract. Answers (:ok reply) or (:error reason)."))

(defgeneric describe-backend (backend)
  (:documentation "BACKEND's metadata plist: :KIND (:protocol or :provider),
:NAME, :SUMMARY and what the kind adds."))

(defclass protocol-backend (backend)
  ((summary :initarg :summary :reader backend-summary)
   (params :initarg :params :initform nil :reader backend-params))
  (:documentation "A backend that speaks one wire shape. Base URL, model and
authentication travel in the request, layered on by a provider."))

(defmethod describe-backend ((backend protocol-backend))
  (list :kind :protocol :name (backend-name backend)
        :summary (backend-summary backend) :params (backend-params backend)))

;;; --- the registry -------------------------------------------------------

(defvar *backends* (make-hash-table :test 'eq))
(defvar *backends-lock* (bt:make-lock))

(defun register-backend (backend &optional (name (backend-name backend)))
  "Make BACKEND answer to the keyword NAME. Answers BACKEND."
  (unless (keywordp name)
    (error "A backend is registered under a keyword, got ~s." name))
  (bt:with-lock-held (*backends-lock*)
    (setf (gethash name *backends*) backend)))

(defun find-backend (name)
  "The backend registered under NAME, or nil."
  (bt:with-lock-held (*backends-lock*)
    (values (gethash name *backends*))))

(defun backends (&key kind)
  "Every registered backend name, sorted. KIND, :protocol or :provider, keeps
the names of that kind."
  (sort (bt:with-lock-held (*backends-lock*)
          (loop for name being the hash-keys of *backends* using (hash-value backend)
                when (or (null kind) (eq kind (getf (describe-backend backend) :kind)))
                  collect name))
        #'string< :key #'symbol-name))

(defun protocols () (backends :kind :protocol))

(defun resolve-backend (designator)
  (etypecase designator
    (backend designator)
    (keyword (find-backend designator))))

;;; --- calling ------------------------------------------------------------

(defun complete (backend &rest request)
  "Perform one turn against BACKEND, a registered keyword or a backend object.
Answers (:ok plist) or (:error reason)."
  (let ((backend (resolve-backend backend))
        (problem (check-request request)))
    (cond
      ((null backend) (fail :unavailable))
      (problem (bad-request "~a" problem))
      (t (handler-case (backend-complete backend request)
           (error (e) (fail (list :error (princ-to-string e)))))))))

(defun complete* (backend &rest request)
  "Like COMPLETE, but answers the reply plist and signals a COMPLETION-ERROR
for a failure."
  (let ((result (apply #'complete backend request)))
    (if (result-error-p result)
        (signal-completion-error (result-error result))
        (second result))))
