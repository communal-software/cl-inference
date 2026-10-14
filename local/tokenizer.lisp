(in-package #:cl-inference/local)

(define-condition tokenizer-error (error)
  ((message :initarg :message :reader tokenizer-error-message))
  (:report (lambda (condition stream)
             (write-string (tokenizer-error-message condition) stream))))

(define-condition unknown-tokenizer (tokenizer-error)
  ((model :initarg :model :reader unknown-tokenizer-model)
   (pre :initarg :pre :reader unknown-tokenizer-pre)))

(defun tokenizer-fail (condition-type control &rest args)
  (error condition-type :message (apply #'format nil control args)))

(defclass tokenizer-class (section-class) ())

(defclass tokenizer ()
  ((format :initarg :format :initform nil :reader tokenizer-format))
  (:metaclass tokenizer-class))

;;; Sections are plists of alists keyed by entry name:
;;;   :pre-tokenizer (kind . options)
;;;   :formats (format-key :options alist)

(defun parse-pre-rule (form)
  (unless (and (consp form) (symbolp (first form)) (evenp (length (rest form)))
               (loop for key in (rest form) by #'cddr always (keywordp key)))
    (tokenizer-fail 'tokenizer-error "Bad pre-tokenizer rule ~S, expected (kind :option value ...)"
                    form))
  (cons (intern (symbol-name (first form)) :keyword) (rest form)))

(defun parse-tokenizer-format (key entries)
  (unless (keywordp key)
    (tokenizer-fail 'tokenizer-error "Bad format key ~S" key))
  (let (options)
    (dolist (entry entries)
      (flet ((bad () (tokenizer-fail 'tokenizer-error "Bad format entry ~S" entry)))
        (unless (and (consp entry) (keywordp (first entry))) (bad))
        (destructuring-bind (head &rest args) entry
          (case head
            (:model (unless (and (= (length args) 1) (stringp (first args))) (bad))
             (push (cons head (first args)) options))
            (:pre (unless (and args (every #'stringp args)) (bad))
             (push (cons head args) options))
            (t (unless (= (length args) 1) (bad))
             (push (cons head (first args)) options))))))
    (list key :options (nreverse options))))

(defun parse-tokenizer-sections (forms)
  (let (rules formats)
    (dolist (form forms)
      (unless (consp form)
        (tokenizer-fail 'tokenizer-error "Bad tokenizer section ~S" form))
      (case (first form)
        (:pre-tokenizer (setf rules (append rules (mapcar #'parse-pre-rule (rest form)))))
        (:format (setf formats (add-format formats (parse-tokenizer-format (second form)
                                                                           (cddr form)))))
        (t (tokenizer-fail 'tokenizer-error "Unknown tokenizer section ~S" (first form)))))
    (list :pre-tokenizer rules :formats formats)))

(defmacro define-tokenizer (name (&rest parents) &body sections)
  "Define tokenizer NAME inheriting from PARENTS. Sections: (:pre-tokenizer (kind . options)...)
and (:format key (:model \"name\") (:pre \"alias\"...)); a child overrides entries by name."
  (let ((spec (parse-tokenizer-sections sections)))
    `(progn
       (defclass ,name ,(or parents '(tokenizer)) () (:metaclass tokenizer-class))
       (setf (class-sections (find-class ',name)) ',spec)
       ',name)))

(defun tokenizer-class-of (designator)
  (let ((class (etypecase designator
                 (symbol (find-class designator))
                 (class designator)
                 (tokenizer (class-of designator)))))
    (mop:ensure-finalized class)
    class))

(defun tokenizer-format-option (tokenizer format option &optional default)
  "Value of OPTION in the FORMAT section, and whether it was declared."
  (let ((cell (assoc option (format-entries (tokenizer-class-of tokenizer) format :options))))
    (if cell (values (cdr cell) t) (values default nil))))

(defun tokenizer-pre-rules (tokenizer)
  "Alist of (kind . options): the pre-tokenizer rules of TOKENIZER (a class, name or instance)."
  (merged-entries (tokenizer-class-of tokenizer) (lambda (sections) (getf sections :pre-tokenizer))))

;;; Selecting and loading

(defgeneric tokenizer-selector (format weights)
  (:documentation "The (values model pre) names WEIGHTS declares, matched against :model and :pre.")
  (:method (format weights)
    (declare (ignore weights))
    (tokenizer-fail 'tokenizer-error "Format ~S carries no tokenizer" format)))

(defgeneric tokenizer-load (tokenizer format weights)
  (:documentation "Fill TOKENIZER from the FORMAT metadata of WEIGHTS."))

(defun find-tokenizer-class (format model pre)
  "The most specific tokenizer whose FORMAT section declares :model MODEL and :pre PRE."
  (find-leaf-class (find-class 'tokenizer)
                   (lambda (class)
                     (and model pre
                          (equal model (tokenizer-format-option class format :model))
                          (member pre (tokenizer-format-option class format :pre) :test #'equal)))
                   (lambda ()
                     (error 'unknown-tokenizer :model model :pre pre
                            :message (format nil "No tokenizer for model ~S, pre ~S in format ~S"
                                             model pre format)))
                   (lambda (leaves)
                     (tokenizer-fail 'tokenizer-error
                                     "Tokenizers ~{~S~^, ~} all declare model ~S, pre ~S for ~S"
                                     (mapcar #'class-name leaves) model pre format))))

(defgeneric tokenizer-class-for-weights (format weights model pre)
  (:method (format weights model pre)
    (declare (ignore weights))
    (find-tokenizer-class format model pre)))

(defun load-tokenizer (weights)
  "Instantiate the tokenizer WEIGHTS declares and load its vocabulary."
  (ensure-open weights)
  (let ((format (weights-format weights)))
    (multiple-value-bind (model pre) (tokenizer-selector format weights)
      (let ((tokenizer (make-instance (tokenizer-class-for-weights format weights model pre) :format format)))
        (tokenizer-load tokenizer format weights)
        tokenizer))))

;; TODO: files without tokenizer.ggml.pre select no tokenizer; llama.cpp uses a default pattern (#32)
(defmethod tokenizer-selector ((format (eql :gguf)) weights)
  (values (weights-metadata weights "tokenizer.ggml.model")
          (weights-metadata weights "tokenizer.ggml.pre")))

;;; Byte-level BPE

(defclass byte-level-bpe (tokenizer)
  ((tokens :reader tokenizer-tokens :documentation "Vocabulary strings, indexed by id.")
   (ids :reader tokenizer-ids :documentation "Token string to id.")
   (octets :reader tokenizer-octets :documentation "Decoded bytes, indexed by id.")
   (types :reader tokenizer-types :documentation "ggml token type, indexed by id.")
   (byte-ids :reader tokenizer-byte-ids :documentation "Token id of each single byte.")
   (merges :reader tokenizer-merges
           :documentation "Fixnum table: left*vocab+right to rank*2^32+merged id.")
   (specials :reader tokenizer-specials
             :documentation "Hash from first character to (string . id), longest first.")
   (special-ids :initform nil :reader tokenizer-special-ids)
   (add-bos :initform nil :reader tokenizer-add-bos-p)
   (ignore-merges :initform nil :reader tokenizer-ignore-merges-p
                  :documentation "Emit a pre-tokenized piece found in the vocabulary as one token.")
   (matchers :reader tokenizer-matchers :documentation "Compiled pre-tokenizer rules."))
  (:metaclass tokenizer-class))

(defconstant +merge-id-bits+ 32)

(defparameter *byte-chars*
  (let ((table (make-array 256 :element-type 'character)) (extra 256))
    (dotimes (byte 256 table)
      (setf (aref table byte)
            (code-char (if (or (<= 33 byte 126) (<= 161 byte 172) (<= 174 byte 255))
                           byte
                           (prog1 extra (incf extra))))))))

(defparameter *char-bytes*
  (let ((table (make-array 512 :element-type 'fixnum :initial-element -1)))
    (dotimes (byte 256 table)
      (setf (aref table (char-code (aref *byte-chars* byte))) byte))))

(defun token-string-octets (token)
  "TOKEN's bytes through the byte-level character map, or NIL when it holds other characters."
  (let ((out (make-array (length token) :element-type '(unsigned-byte 8))))
    (dotimes (i (length token) out)
      (let* ((code (char-code (char token i)))
             (byte (if (< code 512) (aref *char-bytes* code) -1)))
        (when (minusp byte) (return nil))
        (setf (aref out i) byte)))))

(defun utf8-octets (string)
  (coerce (babel:string-to-octets string :encoding :utf-8) '(simple-array (unsigned-byte 8) (*))))

(defconstant +token-normal+ 1)
(defconstant +token-unknown+ 2)
(defconstant +token-control+ 3)
(defconstant +token-user-defined+ 4)

;;; GGUF spells the separator key "seperator_token_id".
(defparameter *special-id-keys*
  '((:bos . "bos_token_id") (:eos . "eos_token_id") (:eot . "eot_token_id")
    (:eom . "eom_token_id") (:pad . "padding_token_id") (:unk . "unknown_token_id")
    (:sep . "seperator_token_id")))

(defun gguf-tokenizer-key (name)
  (concatenate 'string "tokenizer.ggml." name))

(defun load-gguf-vocabulary (weights)
  (flet ((meta (name &optional default)
           (weights-metadata weights (gguf-tokenizer-key name) default)))
    (let ((tokens (meta "tokens")) (types (meta "token_type")) (merges (meta "merges")))
      (unless (and (vectorp tokens) (every #'stringp tokens))
        (tokenizer-fail 'tokenizer-error "~A lacks a list of token strings" (weights-path weights)))
      (unless (and (vectorp merges) (every #'stringp merges))
        (tokenizer-fail 'tokenizer-error "~A lacks a list of merges" (weights-path weights)))
      (when (and types (not (and (vectorp types) (= (length types) (length tokens)))))
        (tokenizer-fail 'tokenizer-error "token_type has ~A entries for ~D tokens"
                        (if (vectorp types) (length types) "no") (length tokens)))
      (values (coerce tokens 'simple-vector)
              (let ((out (make-array (length tokens) :element-type '(unsigned-byte 8)
                                                     :initial-element +token-normal+)))
                (when types
                  (dotimes (i (length tokens))
                    (setf (aref out i) (max 0 (min 255 (aref types i))))))
                out)
              (coerce merges 'simple-vector)))))

(defun build-vocabulary (tokenizer tokens types)
  (let ((ids (make-hash-table :test 'equal :size (length tokens)))
        (octets (make-array (length tokens)))
        (specials (make-hash-table)))
    (dotimes (id (length tokens))
      (let ((token (svref tokens id)) (type (aref types id)))
        (setf (gethash token ids) (or (gethash token ids) id))
        (setf (svref octets id)
              (if (= type +token-normal+)
                  (or (token-string-octets token)
                      (tokenizer-fail 'tokenizer-error "Token ~D ~S is not byte-level text" id token))
                  (or (and (/= type +token-control+) (/= type +token-user-defined+)
                           (token-string-octets token))
                      (utf8-octets token))))
        (when (and (<= +token-unknown+ type +token-user-defined+) (plusp (length token)))
          (push (cons token id) (gethash (char token 0) specials)))))
    (maphash (lambda (char entries)
               (setf (gethash char specials) (sort entries #'> :key (lambda (e) (length (car e))))))
             specials)
    (with-slots ((s-ids ids) (s-octets octets) (s-specials specials)) tokenizer
      (setf s-ids ids s-octets octets s-specials specials))))

(defun build-byte-ids (tokenizer)
  (let ((byte-ids (make-array 256 :element-type '(unsigned-byte 32))))
    (dotimes (byte 256)
      (setf (aref byte-ids byte)
            (or (gethash (string (aref *byte-chars* byte)) (tokenizer-ids tokenizer))
                (tokenizer-fail 'tokenizer-error "Vocabulary has no token for byte ~D" byte))))
    (setf (slot-value tokenizer 'byte-ids) byte-ids)))

(defun build-merges (tokenizer merges)
  (let* ((ids (tokenizer-ids tokenizer)) (vocab (length (tokenizer-tokens tokenizer)))
         (table (make-hash-table :test 'eql :size (length merges))))
    (dotimes (rank (length merges))
      (let* ((merge (svref merges rank)) (space (position #\Space merge)))
        (unless (and space (plusp space) (< space (1- (length merge)))
                     (not (position #\Space merge :start (1+ space))))
          (tokenizer-fail 'tokenizer-error "Malformed merge ~D ~S" rank merge))
        (let ((left (gethash (subseq merge 0 space) ids))
              (right (gethash (subseq merge (1+ space)) ids))
              (result (gethash (remove #\Space merge) ids)))
          (unless (and left right result)
            (tokenizer-fail 'tokenizer-error "Merge ~D ~S names a token absent from the vocabulary"
                            rank merge))
          (let ((key (+ (* left vocab) right)))
            (unless (gethash key table)
              (setf (gethash key table) (+ (ash rank +merge-id-bits+) result)))))))
    (setf (slot-value tokenizer 'merges) table)))

(defun load-gguf-special-ids (tokenizer weights)
  (let ((vocab (length (tokenizer-tokens tokenizer))) ids)
    (loop for (key . name) in *special-id-keys*
          do (multiple-value-bind (id present) (weights-metadata weights (gguf-tokenizer-key name))
               (when present
                 (unless (and (integerp id) (< -1 id vocab))
                   (tokenizer-fail 'tokenizer-error "~A is ~S, outside the ~D-token vocabulary"
                                   name id vocab))
                 (setf (getf ids key) id))))
    (setf (slot-value tokenizer 'special-ids) ids
          (slot-value tokenizer 'add-bos)
          (and (weights-metadata weights (gguf-tokenizer-key "add_bos_token")
                                 (tokenizer-format-option tokenizer :gguf :add-bos))
               t))))

(defmethod tokenizer-load ((tokenizer byte-level-bpe) (format (eql :gguf)) weights)
  (multiple-value-bind (tokens types merges) (load-gguf-vocabulary weights)
    (setf (slot-value tokenizer 'tokens) tokens
          (slot-value tokenizer 'types) types)
    (build-vocabulary tokenizer tokens types)
    (build-byte-ids tokenizer)
    (build-merges tokenizer merges)
    (load-gguf-special-ids tokenizer weights)
    (setf (slot-value tokenizer 'ignore-merges)
          (and (tokenizer-format-option tokenizer :gguf :ignore-merges) t))
    (let ((rules (tokenizer-pre-rules tokenizer)))
      (unless rules
        (tokenizer-fail 'tokenizer-error "~S declares no pre-tokenizer rules"
                        (class-name (class-of tokenizer))))
      (setf (slot-value tokenizer 'matchers) (compile-pre-tokenizer rules)))
    tokenizer))
