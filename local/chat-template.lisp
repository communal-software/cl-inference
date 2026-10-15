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

(defstruct chat-segment
  (text "" :type string)
  (kind :text :type (member :markup :text)))

(defun markup (text)
  (list (make-chat-segment :text text :kind :markup)))

(defun chat-text (text)
  (list (make-chat-segment :text text :kind :text)))

(defun chat-segments-text (segments)
  (with-output-to-string (out)
    (dolist (segment segments)
      (write-string (chat-segment-text segment) out))))

(defun chat-rendering (segments)
  (values (chat-segments-text segments) segments))

(defun message-control-parts (control)
  "Split the two ~A substitutions without interpreting message data as FORMAT syntax."
  (let ((parts nil) (start 0))
    (loop for i from 0 below (length control)
          when (char= (char control i) #\~)
            do (when (= (1+ i) (length control))
                 (chat-template-fail "Incomplete message directive in ~S" control))
               (case (char-upcase (char control (incf i)))
                 (#\A (push (format nil (subseq control start (1- i))) parts)
                       (setf start (1+ i)))
                 ((#\% #\~))
                 (t (chat-template-fail "Message controls support only ~~A, ~~% and ~~~~: ~S"
                                        control))))
    (push (format nil (subseq control start)) parts)
    (unless (= 3 (length parts))
      (chat-template-fail "Message control needs exactly two ~~A directives: ~S" control))
    (nreverse parts)))

(defun parse-chat-template-section (form)
  (flet ((bad () (chat-template-fail "Bad chat template section ~S" form)))
    (unless (and (consp form) (rest form) (every #'stringp (rest form))) (bad))
    (case (first form)
      ((:detect :exclude :stop))
      (:message (unless (= (length form) 2) (bad))
                (message-control-parts (second form)))
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

(defun render-turn (template role body)
  (destructuring-bind (prefix middle suffix)
      (message-control-parts
       (or (first (chat-template-entry template :message))
           (chat-template-fail "~S declares no :message" (class-name (class-of template)))))
    (append (markup prefix) (chat-text role) (markup middle)
            (if (stringp body) (chat-text body) body) (markup suffix))))

(defun render-generation-prompt (template)
  (markup (format nil (or (first (chat-template-entry template :generation-prompt)) ""))))

(defgeneric render-chat (template messages &key tools add-generation-prompt &allow-other-keys)
  (:documentation "Return prompt text and ordered CHAT-SEGMENTs, without automatic BOS.
Only :MARKUP segments permit control-token matching; caller data uses :TEXT.")
  (:method ((template chat-template) messages &key tools add-generation-prompt &allow-other-keys)
    (when tools
      (chat-template-fail "~S cannot render tools" (class-name (class-of template))))
    (chat-rendering
     (append
      (loop for message in messages
            do (when (or (eq (getf message :role) :tool) (getf message :tool-calls))
                 (chat-template-fail "~S cannot render tool messages" (class-name (class-of template))))
            append (render-turn template (message-role-name message) (message-text message)))
      (when add-generation-prompt (render-generation-prompt template))))))

(defun encode-chat-segments (tokenizer prompt segments)
  (let ((out (make-array 0 :element-type '(unsigned-byte 32) :adjustable t :fill-pointer 0))
        (plain-start 0) (offset 0))
    (labels ((emit (ids)
               (loop for id across ids do (vector-push-extend id out)))
             (flush (end)
               (when (< plain-start end)
                 (emit (encode tokenizer (subseq prompt plain-start end) :add-bos nil)))))
      (when (tokenizer-add-bos-p tokenizer)
        (vector-push-extend (or (special-id tokenizer :bos)
                               (tokenizer-fail 'tokenizer-error "Tokenizer has no BOS token"))
                            out))
      (dolist (segment segments)
        (let ((end (+ offset (length (chat-segment-text segment)))))
          (when (eq (chat-segment-kind segment) :markup)
            (loop with position = offset
                  while (< position end)
                  do (multiple-value-bind (id token-end) (special-match tokenizer prompt position t)
                       (if (and id (<= token-end end))
                           (progn
                             (flush position)
                             (vector-push-extend id out)
                             (setf position token-end plain-start token-end))
                           (incf position)))))
          (setf offset end)))
      ;; Keep ordinary spans intact across provenance boundaries for BPE and normalization.
      (flush (length prompt)))
    (coerce out '(simple-array (unsigned-byte 32) (*)))))

(defun encode-chat (template tokenizer messages &rest keys &key &allow-other-keys)
  "Encode template markup as controls and caller data as text. KEYS go to RENDER-CHAT;
the generation prompt defaults to true. BOS follows the tokenizer's policy."
  (let* ((rendered (multiple-value-list
                    (apply #'render-chat template messages (append keys '(:add-generation-prompt t)))))
         (prompt (first rendered)) (segments (second rendered)))
    (unless (and (= 2 (length rendered)) (stringp prompt) (alexandria:proper-list-p segments)
                 (every (lambda (segment) (typep segment 'chat-segment)) segments)
                 (equal prompt (chat-segments-text segments)))
      (chat-template-fail "Renderer must return prompt text and matching chat segments"))
    (encode-chat-segments tokenizer prompt segments)))

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
  "Instantiate the detected template, using architecture fallback only when source is absent."
  (ensure-open weights)
  (let* ((format (weights-format weights))
         (source (chat-template-source format weights))
         (class (if source (detected-chat-template-class source)
                    (architecture-chat-template-class weights format))))
    (unless class
      (error 'unknown-chat-template :source source
                                    :message (format nil "No chat template for ~A~:[, which has no template~;'s template~]"
                                                     (weights-path weights) source)))
    (make-instance class)))
