(in-package #:cl-inference/client)

;;; A provider is data -- a protocol to speak, a base URL, how to authenticate,
;;; a model catalogue and any quirks -- layered under each request before the
;;; call is handed to its protocol.

(defparameter +auth-kinds+ '(:none :bearer :header)
  "The closed set of authentication kinds.")

(defclass provider (backend)
  ((declaration :initarg :declaration :reader provider-declaration)
   (base-url :initarg :base-url :reader provider-base-url)
   (model :initarg :model :initform nil :reader provider-model)
   (api-key :initarg :api-key :initform nil :reader provider-api-key))
  (:documentation "A declaration plus the fields an instance may override:
base URL, model and API key."))

(defun providers () (backends :kind :provider))

(defun provider-protocol (provider)
  (getf (provider-declaration provider) :protocol))

(defun provider-auth (provider)
  (getf (provider-declaration provider) :auth))

(defun provider-defaults (provider)
  (getf (provider-declaration provider) :defaults))

;;; --- the declaration --------------------------------------------------

(defun check-provider-declaration (name declaration)
  "DECLARATION with its :AUTH canonicalised, or an error. Checked when the
provider is defined, so a malformed provider never reaches a call."
  (flet ((problem (format &rest args)
           (error "Provider ~s: ~a" name (apply #'format nil format args))))
    (let ((protocol (getf declaration :protocol))
          (base-url (getf declaration :base-url)))
      (unless (keywordp protocol)
        (problem ":protocol is required, naming a protocol or provider"))
      (unless (named-p base-url)
        (problem ":base-url is required, naming the backend"))
      (unless (http-url-p base-url)
        (problem ":base-url must be an http or https URL"))
      (dolist (key '(:defaults :headers))
        (let ((value (getf declaration key)))
          (unless (and (listp value) (evenp (length value)))
            (problem "~(~a~) must be a plist" key))))
      (unless (listp (getf declaration :models))
        (problem ":models must be a list of model ids"))
      (a:when-let ((extra (set-difference (loop for (key) on declaration by #'cddr
                                                collect key)
                                          '(:protocol :base-url :auth :models
                                            :defaults :headers :summary
                                            :rewrite-request :rewrite-response))))
        (problem "unknown key~p ~{~(~s~)~^, ~}" (length extra) extra))
      (list* :auth (canonical-auth #'problem (or (getf declaration :auth) :none))
             (a:remove-from-plist declaration :auth)))))

(defun canonical-auth (problem auth)
  "AUTH as (:kind k . details), so one GETF reads every kind."
  (let ((kind (if (consp auth) (first auth) auth)))
    (unless (member kind +auth-kinds+)
      (funcall problem ":auth must be one of ~{~(~s~)~^, ~}, got ~s"
               +auth-kinds+ auth))
    (let* ((details (if (eq kind :header) (cddr auth) (and (consp auth) (rest auth))))
           (env (getf details :env))
           (header (and (eq kind :header) (second auth))))
      (when (and (not (eq kind :none)) (not (named-p env)))
        (funcall problem "~(~s~) auth needs :env, naming the variable the key comes from"
                 kind))
      (when (and (eq kind :header) (not (named-p header)))
        (funcall problem ":header auth needs a header name"))
      (case kind
        (:none '(:kind :none))
        (:bearer (list :kind :bearer :env env))
        (:header (list :kind :header :name header :env env))))))

;;; --- credentials ------------------------------------------------------

;;; BYOK: a key comes from the environment variable the declaration names, or
;;; from an :api-key argument. It is never published in metadata.

(defun provider-key (provider)
  "PROVIDER's API key, or NIL when it has none or needs none."
  (let ((auth (provider-auth provider)))
    (unless (eq (getf auth :kind) :none)
      (let ((key (or (provider-api-key provider) (uiop:getenv (getf auth :env)))))
        (when (named-p key) key)))))

(defun provider-key-problem (provider)
  "NIL when PROVIDER can authenticate, else a problem string."
  (let ((auth (provider-auth provider)))
    (unless (or (eq (getf auth :kind) :none) (provider-key provider))
      (format nil "no API key; set ~a or pass :api-key" (getf auth :env)))))

(defun auth-headers (provider)
  "PROVIDER's credential as a header plist."
  (let ((auth (provider-auth provider))
        (key (provider-key provider)))
    (case (getf auth :kind)
      (:bearer (list "authorization" (format nil "Bearer ~a" key)))
      (:header (list (getf auth :name) key)))))

;;; --- layering ---------------------------------------------------------

(defun getf-ci (plist name)
  (loop for (key value) on plist by #'cddr
        when (string-equal key name) return value))

(defun merge-headers (weak strong)
  "WEAK and STRONG as one header plist. A name STRONG carries wins, compared
without case as HTTP does."
  (append (loop for (name value) on weak by #'cddr
                unless (getf-ci strong name)
                  collect name and collect value)
          strong))

(defun layered-request (provider request)
  "REQUEST with PROVIDER's data under it. A plist reads by its first match, so
appending the provider's keys after the caller's is what lets the caller win."
  (let ((headers (merge-headers (merge-headers
                                 (getf (provider-declaration provider) :headers)
                                 (auth-headers provider))
                                (getf request :headers))))
    (append (when headers (list :headers headers))
            (a:remove-from-plist request :headers)
            (list :base-url (provider-base-url provider))
            (when (provider-model provider)
              (list :model (provider-model provider)))
            (provider-defaults provider))))

(defun apply-quirk (provider hook value)
  "VALUE through PROVIDER's HOOK, a quirk the declaration names, or unchanged."
  (a:if-let ((function (getf (provider-declaration provider) hook)))
    (funcall function value)
    value))

;;; --- the exchange -----------------------------------------------------

(defmethod describe-backend ((provider provider))
  (let ((declaration (provider-declaration provider)))
    (list :kind :provider
          :name (backend-name provider)
          :protocol (provider-protocol provider)
          :summary (getf declaration :summary)
          :base-url (provider-base-url provider)
          :models (getf declaration :models)
          :model (provider-model provider)
          ;; The canonical auth form: a kind, a header name and the variable
          ;; the key comes from. Never the key itself.
          :auth (provider-auth provider)
          :defaults (provider-defaults provider)
          :status (if (provider-key-problem provider) :unavailable :ready))))

(defmethod backend-complete ((provider provider) request)
  "Layer PROVIDER's data under REQUEST and hand the call to its protocol,
looked up by name, so a provider may sit over another provider."
  (a:if-let ((problem (provider-key-problem provider)))
    (bad-request "~a" problem)
    (a:if-let ((protocol (find-backend (provider-protocol provider))))
      (apply-quirk provider :rewrite-response
                   (backend-complete protocol
                                     (apply-quirk provider :rewrite-request
                                                  (layered-request provider request))))
      (fail :unavailable))))

;;; --- definition ---------------------------------------------------------

(defun make-provider (name &key base-url model api-key)
  "A new instance of the provider NAME defines, with BASE-URL, MODEL and
API-KEY overriding the declaration."
  (let ((defined (find-backend name)))
    (unless (typep defined 'provider)
      (error "~s is not a defined provider." name))
    (make-instance 'provider
                   :name name
                   :declaration (provider-declaration defined)
                   :base-url (or base-url (provider-base-url defined))
                   :model model
                   :api-key api-key)))

(defmacro define-provider (name &rest declaration
                           &key protocol base-url auth models defaults headers
                                summary rewrite-request rewrite-response)
  "Define and register provider NAME, a keyword. Every value is a form,
evaluated when the definition loads."
  (declare (ignore protocol base-url auth models defaults headers summary
                   rewrite-request rewrite-response))
  (unless (keywordp name)
    (error "DEFINE-PROVIDER: NAME must be a literal keyword."))
  `(register-backend
    (let ((declaration (check-provider-declaration
                        ,name (list ,@(loop for (key form) on declaration by #'cddr
                                            collect key collect form)))))
      (make-instance 'provider :name ,name :declaration declaration
                               :base-url (getf declaration :base-url)))))
