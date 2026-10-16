(in-package #:cl-inference/local)

;;; Hand-written equivalents of the regex alternatives byte-level BPE models split text with,
;;; e.g. llama.cpp's llama3 pattern:
;;;   '[sdtm]|'ll|'re|'ve | [^\r\n\p{L}\p{N}]?\p{L}+ | \p{N}{1,3} | ?[^\s\p{L}\p{N}]+[\r\n]*
;;;   | \s*[\r\n]+ | \s+(?!\S) | \s+
;;; Each rule is a function (text start) returning the end of its match, or NIL.

(deftype text () '(simple-array character (*)))

(declaim (type (simple-array (unsigned-byte 8) (65536)) *char-flags*))
(defvar *char-flags* (make-array 65536 :element-type '(unsigned-byte 8) :initial-element 0)
  "Cache of char-flags for the BMP; zero means not yet computed.")

(defconstant +letter+ 1)
(defconstant +number+ 2)
(defconstant +space+ 4)
(defconstant +known+ 8)
(defconstant +punctuation+ 16)

(defun compute-char-flags (char)
  (let ((category (or (cl-unicode:general-category char) "Cn")))
    (logior +known+
            (if (char= (char category 0) #\L) +letter+ 0)
            (if (char= (char category 0) #\N) +number+ 0)
            (if (char= (char category 0) #\P) +punctuation+ 0)
            (if (cl-unicode:has-property char "White_Space") +space+ 0))))

(declaim (inline char-flags))
(defun char-flags (char)
  (let ((code (char-code char)))
    (if (< code 65536)
        (let ((flags (aref *char-flags* code)))
          (if (zerop flags)
              (setf (aref *char-flags* code) (compute-char-flags char))
              flags))
        (compute-char-flags char))))

(declaim (inline letter-p number-p space-p newline-p))
(defun letter-p (char) (logtest (char-flags char) +letter+))
(defun number-p (char) (logtest (char-flags char) +number+))
(defun space-p (char) (logtest (char-flags char) +space+))
(defun newline-p (char) (or (char= char #\Return) (char= char #\Newline)))

(defgeneric rule-matcher (kind options)
  (:documentation "A function (text start) returning the end of a match of pre-tokenizer rule KIND
at START, or NIL. KIND is a keyword and OPTIONS the rule's plist.")
  (:method (kind options)
    (declare (ignore options))
    (tokenizer-fail 'tokenizer-error "Unknown pre-tokenizer rule ~S" kind)))

(defun check-rule-options (kind options allowed)
  (loop for key in options by #'cddr
        unless (member key allowed)
          do (tokenizer-fail 'tokenizer-error "Rule ~S has no option ~S" kind key)))

(defmethod rule-matcher ((kind (eql :contractions)) options)
  (check-rule-options kind options '(:case-insensitive))
  (let ((fold (if (getf options :case-insensitive) #'char-downcase #'identity)))
    (lambda (text start)
      (declare (type text text) (fixnum start))
      (let ((n (length text)))
        (flet ((at (offset) (and (< (+ start offset) n) (funcall fold (schar text (+ start offset))))))
          (when (eql (at 0) #\')
            (case (at 1)
              ((#\s #\t #\m #\d) (+ start 2))
              (#\r (and (eql (at 2) #\e) (+ start 3)))
              (#\v (and (eql (at 2) #\e) (+ start 3)))
              (#\l (and (eql (at 2) #\l) (+ start 3))))))))))

(defmethod rule-matcher ((kind (eql :letters)) options)
  (check-rule-options kind options '(:prefix))
  (let ((prefix (getf options :prefix)))
    (unless (member prefix '(nil :space :non-letter-non-newline))
      (tokenizer-fail 'tokenizer-error "Rule :letters has no prefix ~S" prefix))
    (lambda (text start)
      (declare (type text text) (fixnum start))
      (let* ((n (length text))
             (from (if (and prefix (< start n)
                            (let ((char (schar text start)))
                              (if (eq prefix :space)
                                  (char= char #\Space)
                                  (not (or (newline-p char) (letter-p char) (number-p char))))))
                       (1+ start)
                       start))
             (end from))
        (declare (fixnum n from end))
        (loop while (and (< end n) (letter-p (schar text end))) do (incf end))
        (and (> end from) end)))))

(defmethod rule-matcher ((kind (eql :digits)) options)
  (check-rule-options kind options '(:max :min :prefix :ascii-only))
  (let ((limit (getf options :max 1)) (minimum (getf options :min 1)) (prefix (getf options :prefix))
        (ascii-only (getf options :ascii-only)))
    (unless (or (null limit) (typep limit '(integer 1)))
      (tokenizer-fail 'tokenizer-error "Rule :digits needs a positive :max or NIL, not ~S" limit))
    (unless (and (typep minimum '(integer 1)) (or (null limit) (<= minimum limit)))
      (tokenizer-fail 'tokenizer-error "Rule :digits has invalid :min ~S for :max ~S" minimum limit))
    (unless (member prefix '(nil :space))
      (tokenizer-fail 'tokenizer-error "Rule :digits has no prefix ~S" prefix))
    (lambda (text start)
      (declare (type text text) (fixnum start))
      (let* ((from (if (and prefix (< start (length text)) (char= (schar text start) #\Space))
                       (1+ start) start))
             (end from) (stop (if limit (min (length text) (+ from limit)) (length text))))
        (declare (fixnum from end stop))
        (loop while (and (< end stop)
                         (if ascii-only
                             (char<= #\0 (schar text end) #\9)
                             (number-p (schar text end))))
              do (incf end))
        (and (>= (- end from) minimum) end)))))

(defmethod rule-matcher ((kind (eql :punctuation-symbols)) options)
  (check-rule-options kind options '())
  (lambda (text start)
    (declare (type text text) (fixnum start))
    (let ((end start))
      (loop while (and (< end (length text))
                       (let ((char (schar text end)))
                         (or (logtest (char-flags char) +punctuation+)
                             (find char "$+<=>^~|"))))
            do (incf end))
      (and (> end start) end))))

(defmethod rule-matcher ((kind (eql :punctuation)) options)
  (check-rule-options kind options '(:leading-space :trailing-newlines))
  (let ((leading-space (getf options :leading-space))
        (trailing-newlines (getf options :trailing-newlines)))
    (lambda (text start)
      (declare (type text text) (fixnum start))
      (let* ((n (length text))
             (from (if (and leading-space (< start n) (char= (schar text start) #\Space))
                       (1+ start)
                       start))
             (end from))
        (declare (fixnum n from end))
        (loop while (and (< end n)
                         (let ((char (schar text end)))
                           (not (or (space-p char) (letter-p char) (number-p char)))))
              do (incf end))
        (when (> end from)
          (when trailing-newlines
            (loop while (and (< end n) (newline-p (schar text end))) do (incf end)))
          end)))))

(defun whitespace-run-end (text start)
  (declare (type text text) (fixnum start))
  (let ((end start) (n (length text)))
    (declare (fixnum end n))
    (loop while (and (< end n) (space-p (schar text end))) do (incf end))
    end))

(defmethod rule-matcher ((kind (eql :newlines)) options)
  "\\s*[\\r\\n]+: whitespace up to and including its last newline."
  (check-rule-options kind options '())
  (lambda (text start)
    (declare (type text text) (fixnum start))
    (let ((run-end (whitespace-run-end text start)))
      (loop for i of-type fixnum from (1- run-end) downto start
            when (newline-p (schar text i)) return (1+ i)))))

(defmethod rule-matcher ((kind (eql :whitespace)) options)
  "\\s+(?!\\S) then \\s+: a run keeps its last character for the next piece when text follows."
  (check-rule-options kind options '())
  (lambda (text start)
    (declare (type text text) (fixnum start))
    (let ((run-end (whitespace-run-end text start)))
      (declare (fixnum run-end))
      (cond ((= run-end start) nil)
            ((or (= run-end (length text)) (= (1+ start) run-end)) run-end)
            (t (1- run-end))))))

(defun compile-pre-tokenizer (rules)
  "RULES is an alist of (kind . options); returns a simple-vector of matcher functions."
  (coerce (loop for (kind . options) in rules collect (rule-matcher kind options)) 'simple-vector))

(defstruct pre-tokenizer-stages matchers)

(defun initialize-pre-tokenizer (tokenizer)
  (let ((rules (tokenizer-pre-rules tokenizer)) (stages (tokenizer-pre-stages tokenizer)))
    (when (and rules stages)
      (tokenizer-fail 'tokenizer-error "~S mixes flat and staged pre-tokenizer rules"
                      (class-name (class-of tokenizer))))
    (unless (or rules stages)
      (tokenizer-fail 'tokenizer-error "~S declares no pre-tokenizer rules"
                      (class-name (class-of tokenizer))))
    (setf (slot-value tokenizer 'matchers)
          (if stages
              (make-pre-tokenizer-stages
               :matchers (map 'vector (lambda (stage) (compile-pre-tokenizer (cdr stage))) stages))
              (compile-pre-tokenizer rules)))))

(defun coerce-text (string)
  (check-type string string)
  (if (typep string 'text) string (coerce string 'text)))

(defun map-flat-pre-pieces (function matchers text)
  "Call FUNCTION with the start and end of each piece of TEXT."
  (declare (type text text) (simple-vector matchers) (function function))
  (let ((position 0) (end (length text)))
    (declare (fixnum position end))
    (loop while (< position end)
          do (let ((next (loop for matcher across matchers
                               for match = (funcall (the function matcher) text position)
                               when (and match (> (the fixnum match) position)) return match)))
               (unless next
                 (tokenizer-fail 'tokenizer-error "No pre-tokenizer rule matches ~S at ~D"
                                 (char text position) position))
               (funcall function position next)
               (setf position next)))))

(defun map-staged-pre-pieces (function stages text)
  (labels ((split-piece (stage start end)
             (if (= stage (length stages))
                 (funcall function start end)
                 (let* ((piece (subseq text start end)) (size (length piece))
                        (position 0) (unmatched 0) (matchers (svref stages stage)))
                   (flet ((emit (from to)
                            (when (< from to)
                              (split-piece (1+ stage) (+ start from) (+ start to)))))
                     (loop while (< position size)
                           for next = (loop for matcher across matchers
                                            for match = (funcall matcher piece position)
                                            when match return match)
                           do (if next
                                  (progn
                                    (unless (and (integerp next) (< position next) (<= next size))
                                      (tokenizer-fail 'tokenizer-error "Invalid stage match ~S at ~D"
                                                      next position))
                                    (emit unmatched position)
                                    (emit position next)
                                    (setf position next unmatched next))
                                  (incf position)))
                     (emit unmatched size))))))
    (when (plusp (length text))
      (split-piece 0 0 (length text)))))

(defun map-pre-pieces (function matchers text)
  (etypecase matchers
    (simple-vector (map-flat-pre-pieces function matchers text))
    (pre-tokenizer-stages
     (map-staged-pre-pieces function (pre-tokenizer-stages-matchers matchers) text))))

(defun pre-tokenize (tokenizer text)
  "The pieces TEXT splits into before BPE, as a list of strings."
  (let ((text (coerce-text text)) pieces)
    (map-pre-pieces (lambda (start end) (push (subseq text start end) pieces))
                    (tokenizer-matchers tokenizer) text)
    (nreverse pieces)))
