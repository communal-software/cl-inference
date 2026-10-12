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

(defun compute-char-flags (char)
  (let ((category (or (cl-unicode:general-category char) "Cn")))
    (logior +known+
            (if (char= (char category 0) #\L) +letter+ 0)
            (if (char= (char category 0) #\N) +number+ 0)
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
    (unless (member prefix '(nil :non-letter-non-newline))
      (tokenizer-fail 'tokenizer-error "Rule :letters has no prefix ~S" prefix))
    (lambda (text start)
      (declare (type text text) (fixnum start))
      (let* ((n (length text))
             (from (if (and prefix (< start n)
                            (let ((char (schar text start)))
                              (not (or (newline-p char) (letter-p char) (number-p char)))))
                       (1+ start)
                       start))
             (end from))
        (declare (fixnum n from end))
        (loop while (and (< end n) (letter-p (schar text end))) do (incf end))
        (and (> end from) end)))))

(defmethod rule-matcher ((kind (eql :digits)) options)
  (check-rule-options kind options '(:max))
  (let ((limit (getf options :max 1)))
    (unless (typep limit '(integer 1))
      (tokenizer-fail 'tokenizer-error "Rule :digits needs a positive :max, not ~S" limit))
    (lambda (text start)
      (declare (type text text) (fixnum start))
      (let ((end start) (stop (min (length text) (+ start limit))))
        (declare (fixnum end stop))
        (loop while (and (< end stop) (number-p (schar text end))) do (incf end))
        (and (> end start) end)))))

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

(defun coerce-text (string)
  (check-type string string)
  (if (typep string 'text) string (coerce string 'text)))

(defun map-pre-pieces (function matchers text)
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

(defun pre-tokenize (tokenizer text)
  "The pieces TEXT splits into before BPE, as a list of strings."
  (let ((text (coerce-text text)) pieces)
    (map-pre-pieces (lambda (start end) (push (subseq text start end) pieces))
                    (tokenizer-matchers tokenizer) text)
    (nreverse pieces)))
