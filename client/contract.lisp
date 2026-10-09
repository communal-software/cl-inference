(in-package #:cl-inference/client)

;;; The neutral model contract. A request is a plist (:messages, :tools,
;;; :model, sampling parameters, :timeout, :cancel, :stream, :ref); a reply is
;;; (:ok plist) with :role :assistant, :content, :tool-calls, :done and :meta.

(defparameter +roles+ '(:system :user :assistant :tool)
  "The closed set of message roles.")

;;; --- content ----------------------------------------------------------

(defun text-block (text) (list :type :text :text text))

(defun normalize-content (content)
  "CONTENT as a list of typed blocks. A flat string is the degenerate
single-text-block case."
  (etypecase content
    (null nil)
    (string (list (text-block content)))
    (cons content)))

(defun content-text (content)
  "The text blocks of CONTENT, concatenated."
  (with-output-to-string (out)
    (dolist (block (normalize-content content))
      (when (eq (getf block :type) :text)
        (write-string (getf block :text "") out)))))

;;; --- names and keys -----------------------------------------------------

(defun named-p (value)
  (and (stringp value) (plusp (length value))))

(defun http-url-p (url)
  ;; Checked up front so that everything past it failing before a status line
  ;; is a transport failure and nothing else.
  (some (lambda (scheme)
          (and (>= (length url) (length scheme))
               (string-equal scheme url :end2 (length scheme))))
        '("http://" "https://")))

(defun wire-key (name)
  "NAME as a JSON property name: :MAX-TOKENS is max_tokens."
  (substitute #\_ #\- (string-downcase (symbol-name name))))

(defun lisp-key (key)
  (a:make-keyword (string-upcase (substitute #\- #\_ key))))

(defun wire-tool-name (name)
  (string-downcase (symbol-name name)))

(defun lisp-tool-name (name)
  (a:make-keyword (string-upcase name)))

;;; --- values -------------------------------------------------------------

;;; Tool arguments are a plist in the contract and a JSON object on the wire.
;;; The tool's own schema names each value's type, so a map and an object stay
;;; distinguishable where a plist alone leaves them ambiguous.

(defun tool-schema (metadata)
  "METADATA's parameter schema."
  (getf metadata :params))

(defun call-schema (name tools)
  "The schema of the tool NAME names among TOOLS, a list of tool metadata."
  (a:when-let ((metadata (find name tools :key (lambda (m) (getf m :name)))))
    (tool-schema metadata)))

(defun make-call (id name arguments tools)
  "A tool call as a reply carries it. It keeps the schema of the tool it names
among TOOLS, so it renders the same wherever it is replayed."
  (let ((schema (call-schema name tools)))
    (list* :id id :name name :arguments arguments
           (when schema (list :schema schema)))))

(defun call-arguments->json (call tools)
  "CALL's arguments as JSON, by the schema the call carries, else by the tool
TOOLS names."
  (arguments->json (getf call :arguments)
                   (or (getf call :schema)
                       (call-schema (getf call :name) tools))))

(defun arguments->json (arguments schema)
  (let ((json (json-object)))
    (loop for (name value) on arguments by #'cddr
          for param = (find name schema :key #'param-name)
          do (setf (json-get json (wire-key name))
                   (value->json value (and param (param-type param)))))
    json))

(defun value->json (value spec)
  (cond
    ((null spec) (untyped->json value))
    ((spec-is spec "OR") (if (null value) 'null (value->json value (third spec))))
    ((spec-is spec "ARRAY-OF")
     (map 'vector (lambda (element) (value->json element (second spec))) value))
    ((spec-is spec "MAP-OF")
     (let ((json (json-object)))
       (loop for (key entry) on value by #'cddr
             do (setf (json-get json (as-text key))
                      (value->json entry (second spec))))
       json))
    ((spec-is spec "OBJECT") (arguments->json value (rest spec)))
    ((spec-is spec "ANY") (untyped->json value))
    (t (json-value value))))

;;; A call built by hand carries no schema, so it renders by this guess: a plist
;;; of keywords is an object and any other list an array.

(defun untyped->json (value)
  (cond
    ((null value) nil)
    ((keywordp value) (json-value value))
    ((not (consp value)) value)
    ((and (evenp (length value))
          (loop for (name) on value by #'cddr always (keywordp name)))
     (let ((json (json-object)))
       (loop for (name entry) on value by #'cddr
             do (setf (json-get json (wire-key name)) (untyped->json entry)))
       json))
    (t (map 'vector #'untyped->json value))))

(defun json->arguments (json schema)
  "JSON, a parsed object, as an argument plist. Names become keywords, which
is what COERCE-ARGS matches a schema on."
  (let ((plist '()))
    (maphash (lambda (key value)
               (let* ((name (lisp-key key))
                      (param (find name schema :key #'param-name)))
                 (push name plist)
                 (push (json->value value (and param (param-type param))) plist)))
             json)
    (nreverse plist)))

(defun json->value (value spec)
  (cond
    ((eq value 'null) nil)
    ((null spec) (untyped->lisp value))
    ((spec-is spec "OR") (json->value value (third spec)))
    ((spec-is spec "ARRAY-OF")
     (map 'list (lambda (element) (json->value element (second spec))) value))
    ((spec-is spec "MAP-OF")
     (loop for key being the hash-keys of value using (hash-value entry)
           collect key collect (json->value entry (second spec))))
    ((spec-is spec "OBJECT") (json->arguments value (rest spec)))
    ((spec-is spec "ANY") (untyped->lisp value))
    ;; A member arrives as its name and an integer as digits; COERCE-ARGS
    ;; takes both, so a scalar passes through untouched.
    (t value)))

(defun untyped->lisp (value)
  (cond
    ((eq value 'null) nil)
    ((hash-table-p value) (json->arguments value nil))
    ((and (vectorp value) (not (stringp value)))
     (map 'list #'untyped->lisp value))
    (t value)))

;;; --- the tools array ------------------------------------------------------

;;; Each tool's schema renders straight through SCHEMA->JSON-SCHEMA, in the
;;; shape both OpenAI and Ollama's native endpoint use:
;;; {"type":"function","function":{name,description,parameters}}.

(defun tools->json (tools)
  (map 'vector
       (lambda (metadata)
         (let ((function (json-object
                          "name" (wire-tool-name (getf metadata :name))
                          "parameters" (schema->json-schema
                                        (tool-schema metadata)))))
           (a:when-let ((summary (getf metadata :summary)))
             (setf (json-get function "description") summary))
           (json-object "type" "function" "function" function)))
       tools))

;;; --- requests ---------------------------------------------------------

;;; COERCE-ARGS is deliberately not used here: it rejects a key its schema
;;; does not name, and the contract ignores unknown keys so a portable caller
;;; may offer a superset. The checks below are the whole pre-flight.

(defun check-request (request)
  "NIL when REQUEST satisfies the contract, else a problem string."
  (cond
    ((not (and (listp request) (evenp (length request))))
     "request must be a plist")
    ((not (listp (getf request :messages)))
     "messages must be a list")
    ((null (getf request :messages))
     "messages is required")
    ((not (typep (getf request :stream) '(or null function symbol)))
     "stream must be a function")
    (t (some #'check-message (getf request :messages)))))

(defun check-message (message)
  (let ((role (getf message :role)))
    (cond
      ((not (and (listp message) (evenp (length message))))
       (format nil "message must be a plist, got ~s" message))
      ((not (member role +roles+))
       (format nil "role must be one of ~{~(~s~)~^, ~}, got ~s" +roles+ role))
      ((and (eq role :tool) (not (getf message :tool-call-id)))
       "a tool message must carry :tool-call-id")
      (t (some #'check-tool-call (getf message :tool-calls))))))

(defun check-tool-call (call)
  (unless (and (listp call) (getf call :id) (getf call :name))
    (format nil "a tool call must carry :id and :name, got ~s" call)))

;;; --- streaming events ---------------------------------------------------

;;; A neutral event vocabulary, never raw provider chunks. A turn ends with
;;; exactly one :DONE, whose reason is the finish reason or, when the exchange
;;; failed, the failed result.

(defun text-delta (ref text)
  (list :type :text-delta :ref ref :text text))

(defun tool-call-delta (ref &key id name arguments)
  "ARGUMENTS is a fragment of the call's argument text, which arrives split
across deltas."
  (list :type :tool-call-delta :ref ref :id id :name name :arguments arguments))

(defun done (ref &optional reason)
  (list :type :done :ref ref :reason reason))

(defun emit-event (sink event)
  "Deliver EVENT to SINK, a function or nil. A sink that signals is ignored: a
misbehaving consumer never fails the completion it observes."
  (when sink
    (ignore-errors (funcall sink event)))
  event)

(defun done-reason (result)
  (if (result-error-p result)
      result
      (getf (getf (second result) :meta) :finish-reason)))

(defun make-reply (text calls reason meta)
  (list :ok (list :role :assistant
                  :content (when (plusp (length text)) (normalize-content text))
                  :tool-calls calls
                  :done t
                  :meta (list* :finish-reason reason meta))))

(defun reply-prompt-tokens (reply)
  "The prompt's size in tokens as the backend counted it, from REPLY's :META
:USAGE :PROMPT-TOKENS, or nil when it reported none."
  (let ((tokens (getf (getf (getf reply :meta) :usage) :prompt-tokens)))
    (and (integerp tokens) (plusp tokens) tokens)))

(defun finish-reason (value)
  "A wire finish/done reason as a keyword: tool_calls is :TOOL-CALLS."
  (when (stringp value) (lisp-key value)))

(defun text-of (value)
  (if (stringp value) value ""))
