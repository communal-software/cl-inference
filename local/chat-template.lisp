(in-package #:cl-inference/local)

(define-condition chat-template-error (error)
  ((message :initarg :message :reader chat-template-error-message))
  (:report (lambda (condition stream)
             (write-string (chat-template-error-message condition) stream))))

(define-condition unknown-chat-template (chat-template-error)
  ((source :initarg :source :reader unknown-chat-template-source)))

(defun chat-template-fail (control &rest args)
  (error 'chat-template-error :message (apply #'format nil control args)))

(defclass chat-template-class (section-class) ())

(defclass chat-template ()
  ()
  (:metaclass chat-template-class))

;;; Sections are a plist with one alist of entries, keyed by section name:
;;;   :entries (:detect marker...) (:message control) (:generation-prompt control) (:stop token...)

(defun parse-chat-template-section (form)
  (flet ((bad () (chat-template-fail "Bad chat template section ~S" form)))
    (unless (and (consp form) (rest form) (every #'stringp (rest form))) (bad))
    (case (first form)
      ((:detect :exclude :stop))
      (:message (unless (and (= (length form) 2)
                             (handler-case (format nil (second form) "role" "text")
                               (error () nil)))
                  (bad)))
      (:generation-prompt (unless (and (= (length form) 2)
                                       (handler-case (format nil (second form))
                                         (error () nil)))
                            (bad)))
      (t (chat-template-fail "Unknown chat template section ~S" (first form))))
    form))

(defun parse-chat-template-sections (forms)
  (let (entries)
    (dolist (form forms)
      (parse-chat-template-section form)
      (when (assoc (first form) entries)
        (chat-template-fail "Duplicate chat template section ~S" (first form)))
      (push form entries))
    (list :entries (nreverse entries))))

(defmacro define-chat-template (name (&rest parents) &body sections)
  "Define chat template NAME inheriting from PARENTS. Sections: (:detect \"marker\"...),
(:exclude \"marker\"...),
(:message \"control\"), (:generation-prompt \"control\") and (:stop \"token\"...). A child
overrides sections by name. Messages that need more than the plain turn format specialise
`render-chat`."
  (let ((spec (parse-chat-template-sections sections)))
    `(progn
       (defclass ,name ,(or parents '(chat-template)) () (:metaclass chat-template-class))
       (setf (class-sections (find-class ',name)) ',spec)
       ',name)))

(defun chat-template-class-of (designator)
  (let ((class (etypecase designator
                 (symbol (find-class designator))
                 (class designator)
                 (chat-template (class-of designator)))))
    (mop:ensure-finalized class)
    class))

(defun chat-template-entry (template name)
  "The arguments of section NAME of TEMPLATE (a class, name or instance)."
  (cdr (assoc name (merged-entries (chat-template-class-of template)
                                   (lambda (sections) (getf sections :entries))))))

;;; Rendering

(defun message-role-name (message)
  (string-downcase (symbol-name (getf message :role))))

(defun message-text (message)
  (ci:content-text (getf message :content)))

(defun render-turn (template role text)
  "One turn of ROLE (a string) with TEXT, in TEMPLATE's message format."
  (format nil (or (first (chat-template-entry template :message))
                  (chat-template-fail "~S declares no :message" (class-name (class-of template))))
          role text))

(defun render-generation-prompt (template)
  (format nil (or (first (chat-template-entry template :generation-prompt)) "")))

(defgeneric render-chat (template messages &key tools add-generation-prompt &allow-other-keys)
  (:documentation "MESSAGES, client message plists, as the prompt text TEMPLATE defines. TOOLS
are client tool metadata. The text carries no BOS token; `encode` adds it.")
  (:method ((template chat-template) messages &key tools add-generation-prompt &allow-other-keys)
    (when tools
      (chat-template-fail "~S cannot render tools" (class-name (class-of template))))
    (with-output-to-string (out)
      (dolist (message messages)
        (when (or (eq (getf message :role) :tool) (getf message :tool-calls))
          (chat-template-fail "~S cannot render tool messages" (class-name (class-of template))))
        (write-string (render-turn template (message-role-name message) (message-text message))
                      out))
      (when add-generation-prompt
        (write-string (render-generation-prompt template) out)))))

;; TODO: parse tool calls and reasoning out of a reply (#36)
;; TODO: user text that spells a control token such as <|im_end|> encodes as that token (#37)
(defun encode-chat (template tokenizer messages &rest keys &key &allow-other-keys)
  "MESSAGES rendered by TEMPLATE and encoded with its control tokens as single ids. KEYS go to
`render-chat`; the generation prompt is added unless `:add-generation-prompt nil`."
  (encode tokenizer
          (apply #'render-chat template messages (append keys '(:add-generation-prompt t)))
          :special t))

(defun chat-template-stop-ids (template tokenizer)
  "The ids that end a reply: TEMPLATE's :stop tokens present in the vocabulary, and the
end-of-sequence, end-of-turn and end-of-message ids the file declares."
  (remove-duplicates
   (append (loop for token in (chat-template-entry template :stop)
                 for id = (token-id tokenizer token)
                 when id collect id)
           (loop for key in '(:eos :eot :eom)
                 for id = (tokenizer-special-id tokenizer key)
                 when id collect id))
   :from-end t))

;;; Selecting and loading

(defgeneric chat-template-source (format weights)
  (:documentation "The chat template text WEIGHTS carries, or NIL.")
  (:method (format weights)
    (declare (ignore format weights))
    nil))

(defmethod chat-template-source ((format (eql :gguf)) weights)
  (let ((source (weights-metadata weights "tokenizer.chat_template")))
    (and (stringp source) source)))

(defun detected-chat-template-class (source)
  "The most specific chat template whose :detect markers all occur in SOURCE and :exclude
markers none do."
  (find-leaf-class (find-class 'chat-template)
                   (lambda (class)
                     (let ((markers (chat-template-entry class :detect)))
                       (and markers
                            (every (lambda (marker) (search marker source)) markers)
                            (notany (lambda (marker) (search marker source))
                                    (chat-template-entry class :exclude)))))
                   (constantly nil)
                   (lambda (leaves)
                     (chat-template-fail "Chat templates ~{~S~^, ~} all match the file's template"
                                         (mapcar #'class-name leaves)))))

(defun architecture-chat-template-class (weights format)
  "The chat template WEIGHTS's architecture names in its FORMAT section, or NIL."
  (let* ((architecture (handler-case (find-architecture-class format
                                                              (weights-architecture-name weights))
                         (unknown-architecture () nil)))
         (name (and architecture
                    (architecture-format-option architecture format :chat-template))))
    (when name
      (or (find (symbol-name name) (class-subclasses (find-class 'chat-template))
                :key (lambda (class) (symbol-name (class-name class))) :test #'string=)
          (chat-template-fail "~S names the unknown chat template ~S"
                              (class-name architecture) name)))))

(defun load-chat-template (weights)
  "Instantiate the chat template WEIGHTS calls for: the one whose markers occur in the file's
template, else the one its architecture names with `(:chat-template name)`."
  (ensure-open weights)
  (let* ((format (weights-format weights))
         (source (chat-template-source format weights))
         (class (or (and source (detected-chat-template-class source))
                    (architecture-chat-template-class weights format))))
    (unless class
      (error 'unknown-chat-template :source source
                                    :message (format nil "No chat template for ~A~:[, which has no template~;'s template~]"
                                                     (weights-path weights) source)))
    (make-instance class)))
