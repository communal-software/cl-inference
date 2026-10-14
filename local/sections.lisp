(in-package #:cl-inference/local)

;;; Machinery shared by the class-based DSLs (architectures, tokenizers). A DSL class keeps its
;;; parsed sections as plists of alists keyed by entry name; inheritance merges them by name.

(defclass section-class (standard-class)
  ((sections :initform nil :accessor class-sections)))

(defmethod mop:validate-superclass ((class section-class) (superclass standard-class))
  t)

(defun class-lineage (class)
  "CLASS and its ancestors of the same metaclass, most general first."
  (mop:ensure-finalized class)
  (let ((metaclass (class-of class)))
    (reverse (remove-if-not (lambda (c) (typep c metaclass))
                            (mop:class-precedence-list class)))))

(defun same-key-p (a b)
  "Keys match by symbol name, so a child in another package overrides what it inherits."
  (or (eql a b)
      (and (symbolp a) (symbolp b) (string= (symbol-name a) (symbol-name b)))))

(defstruct section-removal name)

(defmethod make-load-form ((removal section-removal) &optional environment)
  (make-load-form-saving-slots removal :environment environment))

(defun merge-section-entries (entries result)
  (dolist (entry entries result)
    (if (section-removal-p entry)
        (setf result (remove (section-removal-name entry) result
                             :key #'car :test #'same-key-p))
        (let ((cell (assoc (car entry) result :test #'same-key-p)))
          (if cell
              (setf (cdr cell) (cdr entry))
              (setf result (append result (list (cons (car entry) (cdr entry))))))))))

(defun merged-entries (class reader)
  "Merge entries in class precedence order; removals discard matching names."
  (let (result)
    (dolist (ancestor (class-lineage class) result)
      (setf result (merge-section-entries
                    (funcall reader (class-sections ancestor)) result)))))

(defun format-entries (class format kind)
  (merged-entries class (lambda (sections)
                          (getf (rest (assoc format (getf sections :formats))) kind))))

(defun merge-format (existing new)
  "Combine two (key :kind entries ...) sections for one format key."
  (if existing
      (list* (first existing)
             (loop with kinds = (remove-duplicates
                                 (append (loop for k in (rest existing) by #'cddr collect k)
                                         (loop for k in (rest new) by #'cddr collect k)))
                   for kind in kinds
                   append (list kind (append (getf (rest existing) kind)
                                             (getf (rest new) kind)))))
      new))

(defun add-format (formats new)
  "FORMATS with the section NEW added, merged into any section for the same key."
  (append (remove (first new) formats :key #'first)
          (list (merge-format (assoc (first new) formats) new))))

(defun class-subclasses (class)
  (let ((children (mop:class-direct-subclasses class)))
    (remove-duplicates (append children (mapcan #'class-subclasses children)))))

(defun find-leaf-class (root matchp on-none on-many)
  "The most specific subclass of ROOT satisfying MATCHP. ON-NONE is called with no arguments
when nothing matches, ON-MANY with the tied classes when several leaves match."
  (let* ((matches (remove-if-not matchp (class-subclasses root)))
         (leaves (remove-if (lambda (class)
                              (some (lambda (other) (and (not (eq other class))
                                                         (subtypep other class)))
                                    matches))
                            matches)))
    (cond ((null leaves) (funcall on-none))
          ((rest leaves) (funcall on-many leaves))
          (t (first leaves)))))
