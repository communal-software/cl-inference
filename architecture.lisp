(in-package #:cl-inference)

(define-condition architecture-error (error)
  ((message :initarg :message :reader architecture-error-message))
  (:report (lambda (condition stream)
             (write-string (architecture-error-message condition) stream))))

(define-condition unknown-architecture (architecture-error)
  ((name :initarg :name :reader unknown-architecture-name)))

(defun architecture-fail (condition-type control &rest args)
  (error condition-type :message (apply #'format nil control args)))

(defclass architecture-class (standard-class)
  ((sections :initform nil :accessor architecture-class-sections)))

(defmethod mop:validate-superclass ((class architecture-class) (superclass standard-class))
  t)

(defclass architecture ()
  ((format :initarg :format :initform nil :reader architecture-format))
  (:metaclass architecture-class))

;;; Sections are plists of alists keyed by entry name:
;;;   :hparams (name . plist)  :blocks (slot class . initargs)
;;;   :formats (format-key :options alist :hparam-keys alist :tensors alist)

(defun parse-hparam (form)
  (destructuring-bind (name &rest plist) (if (consp form) form (list form))
    (unless (and (symbolp name) (null (set-difference (loop for key in plist by #'cddr
                                                            collect key)
                                                       '(:default))))
      (architecture-fail 'architecture-error "Bad hparam ~S" form))
    (cons name plist)))

(defun parse-block (form)
  (unless (and (consp form) (symbolp (first form)) (second form) (symbolp (second form)))
    (architecture-fail 'architecture-error "Bad block ~S, expected (slot class . initargs)" form))
  form)

(defun parse-format-entry (entry options hparam-keys tensors)
  (flet ((bad () (architecture-fail 'architecture-error "Bad format entry ~S" entry)))
    (unless (and (consp entry) (keywordp (first entry))) (bad))
    (destructuring-bind (head &rest args) entry
      (case head
        (:hparam (unless (and (= (length args) 2) (symbolp (first args)) (stringp (second args)))
                   (bad))
         (push (cons (first args) (second args)) hparam-keys))
        (:tensor (unless (and (>= (length args) 2) (symbolp (first args)) (stringp (second args))
                              (evenp (length args))
                              (null (set-difference (loop for key in (cddr args) by #'cddr
                                                          collect key)
                                                    '(:optional))))
                   (bad))
         (push (cons (first args) (rest args)) tensors))
        (t (unless (= (length args) 1) (bad))
         (push (cons head (first args)) options))))
    (values options hparam-keys tensors)))

(defun parse-format (key entries)
  (unless (keywordp key)
    (architecture-fail 'architecture-error "Bad format key ~S" key))
  (let (options hparam-keys tensors)
    (dolist (entry entries)
      (setf (values options hparam-keys tensors)
            (parse-format-entry entry options hparam-keys tensors)))
    (list key :options (nreverse options) :hparam-keys (nreverse hparam-keys)
              :tensors (nreverse tensors))))

(defun merge-format (existing new)
  (if existing
      (list* (first existing)
             (loop for kind in '(:options :hparam-keys :tensors)
                   append (list kind (append (getf (rest existing) kind)
                                             (getf (rest new) kind)))))
      new))

(defun parse-sections (forms)
  (let (hparams blocks formats)
    (dolist (form forms)
      (unless (consp form)
        (architecture-fail 'architecture-error "Bad architecture section ~S" form))
      (case (first form)
        (:hparams (setf hparams (append hparams (mapcar #'parse-hparam (rest form)))))
        (:blocks (setf blocks (append blocks (mapcar #'parse-block (rest form)))))
        (:format (let ((new (parse-format (second form) (cddr form))))
                   (setf formats (append (remove (first new) formats :key #'first)
                                         (list (merge-format (assoc (first new) formats)
                                                             new))))))
        (t (architecture-fail 'architecture-error "Unknown architecture section ~S"
                              (first form)))))
    (list :hparams hparams :blocks blocks :formats formats)))

(defun hparam-slot (hparam)
  (let ((name (first hparam)))
    `(,name :initarg ,(intern (symbol-name name) :keyword))))

(defmacro define-architecture (name (&rest parents) &body sections)
  "Define architecture NAME inheriting from PARENTS. Sections: (:hparams ...),
(:blocks ...), and (:format key ...); a child overrides entries by name."
  (let ((spec (parse-sections sections)))
    `(progn
       (defclass ,name ,(or parents '(architecture))
         ,(mapcar #'hparam-slot (getf spec :hparams))
         (:metaclass architecture-class))
       (setf (architecture-class-sections (find-class ',name)) ',spec)
       (check-architecture (find-class ',name))
       ',name)))

(defun architecture-class-of (designator)
  (let ((class (etypecase designator
                 (symbol (find-class designator))
                 (class designator)
                 (architecture (class-of designator)))))
    (mop:ensure-finalized class)
    class))

(defun architecture-lineage (class)
  "CLASS and its ancestors that are architectures, most general first."
  (mop:ensure-finalized class)
  (reverse (remove-if-not (lambda (c) (typep c 'architecture-class))
                          (mop:class-precedence-list class))))

(defun merged-entries (class reader)
  "Entries from READER over CLASS's lineage, parents first; a later entry replaces an earlier
one with the same key in place."
  (let (result)
    (dolist (ancestor (architecture-lineage class) result)
      (dolist (entry (funcall reader (architecture-class-sections ancestor)))
        (let ((cell (assoc (car entry) result)))
          (if cell
              (setf (cdr cell) (cdr entry))
              (setf result (append result (list (cons (car entry) (cdr entry)))))))))))

(defun format-entries (class format kind)
  (merged-entries class (lambda (sections)
                          (getf (rest (assoc format (getf sections :formats))) kind))))

(defun check-architecture (class)
  (let ((hparams (mapcar #'car (merged-entries class (lambda (s) (getf s :hparams))))))
    (loop for (format . spec) in (getf (architecture-class-sections class) :formats)
          do (loop for (name) in (getf spec :hparam-keys)
                   unless (member name hparams)
                     do (architecture-fail 'architecture-error
                                           "~S maps ~S for format ~S, which is not a declared hparam"
                                           (class-name class) name format)))))

(defun architecture-hparam-specs (architecture)
  "Alist of (name . plist) for every hparam of ARCHITECTURE (a class, name or instance)."
  (merged-entries (architecture-class-of architecture) (lambda (s) (getf s :hparams))))

(defun architecture-block-slots (architecture)
  (mapcar #'car (merged-entries (architecture-class-of architecture)
                                (lambda (s) (getf s :blocks)))))

(defun architecture-block-spec (architecture slot)
  "The (class . initargs) declared for block SLOT, or NIL."
  (cdr (assoc slot (merged-entries (architecture-class-of architecture)
                                   (lambda (s) (getf s :blocks))))))

(defun architecture-format-option (architecture format option &optional default)
  "Value of OPTION in the FORMAT section, and whether it was declared."
  (let ((cell (assoc option (format-entries (architecture-class-of architecture) format
                                            :options))))
    (if cell (values (cdr cell) t) (values default nil))))

(defun hparam (architecture name)
  (if (slot-exists-p architecture name)
      (slot-value architecture name)
      (architecture-fail 'architecture-error "~S has no hparam ~S"
                         (class-name (class-of architecture)) name)))

(defun architecture-arch-name (class format)
  (values (architecture-format-option class format :arch)))

(defun architecture-tensor-spec (architecture role)
  (let ((format (architecture-format architecture)))
    (or (cdr (assoc role (format-entries (architecture-class-of architecture) format :tensors)))
        (architecture-fail 'architecture-error "~S has no tensor ~S for format ~S"
                           (class-name (class-of architecture)) role format))))

(defun architecture-tensor-name (architecture role &optional layer)
  "The file's name for tensor ROLE; LAYER fills ~d in per-layer templates."
  (let ((template (first (architecture-tensor-spec architecture role))))
    (if layer (format nil template layer) (format nil template))))

(defun architecture-tensor (architecture weights role &optional layer)
  "Tensor ROLE as a view, or NIL when it is optional and absent."
  (let* ((name (architecture-tensor-name architecture role layer))
         (optional (getf (rest (architecture-tensor-spec architecture role)) :optional)))
    (if (and optional (null (weights-tensor-info-ref (ensure-open weights) name)))
        nil
        (weights-tensor weights name))))

(defun architecture-subclasses (class)
  (let ((children (mop:class-direct-subclasses class)))
    (remove-duplicates (append children (mapcan #'architecture-subclasses children)))))

(defun find-architecture-class (format name)
  "The most specific architecture whose FORMAT section declares :arch NAME."
  (let* ((matches (remove-if-not (lambda (class) (and name (equal name (architecture-arch-name class format))))
                                 (architecture-subclasses (find-class 'architecture))))
         (leaves (remove-if (lambda (class)
                              (some (lambda (other) (and (not (eq other class))
                                                         (subtypep other class)))
                                    matches))
                            matches)))
    (cond ((null leaves)
           (architecture-fail 'unknown-architecture "No architecture for ~S in format ~S"
                              name format))
          ((rest leaves)
           (architecture-fail 'architecture-error "Architectures ~{~S~^, ~} all declare ~S for ~S"
                              (mapcar #'class-name leaves) name format))
          (t (first leaves)))))

(defun hparam-initargs (class weights format arch-name)
  (let ((keys (format-entries class format :hparam-keys)))
    (loop for (name . options) in (architecture-hparam-specs class)
          for template = (cdr (assoc name keys))
          for (value present) = (multiple-value-list
                                 (if template
                                     (weights-metadata weights (format nil template arch-name))
                                     (values nil nil)))
          unless (or present (member :default options))
            do (architecture-fail 'architecture-error "~S needs hparam ~S, missing from ~A"
                                  (class-name class) name (weights-path weights))
          append (list (intern (symbol-name name) :keyword)
                       (if present value (getf options :default))))))

(defun load-architecture (weights)
  "Instantiate the architecture WEIGHTS declares, with its hparams read from metadata."
  (ensure-open weights)
  (let* ((format (weights-format weights))
         (arch-name (weights-architecture-name weights))
         (class (find-architecture-class format arch-name)))
    (apply #'make-instance class :format format
           (hparam-initargs class weights format arch-name))))
