(in-package #:cl-inference/local)

;;; JSON text as Python's json.dumps writes it with ensure_ascii=False, which is how
;;; transformers' tojson filter renders tool definitions and arguments in chat templates.

(defun shortest-float-digits (x)
  "The shortest digits that read back as the positive float X, and the exponent e with
X = d.ddd * 10^e."
  (let* ((text (let ((*read-default-float-format* (type-of x))) (prin1-to-string x)))
         (epos (position #\e text :test #'char-equal))
         (mantissa (subseq text 0 epos))
         (dot (position #\. mantissa))
         (digits (remove #\. mantissa))
         (leading (or (position #\0 digits :test #'char/=) 0)))
    (values (string-right-trim "0" (subseq digits leading))
            (- (+ (if epos (parse-integer text :start (1+ epos)) 0) dot) leading 1))))

(defun python-float (x)
  "X as Python's repr writes a float."
  (if (zerop x)
      (if (minusp (float-sign x)) "-0.0" "0.0")
      (multiple-value-bind (digits e) (shortest-float-digits (abs x))
        (format nil "~:[~;-~]~A" (minusp x)
                (cond ((not (<= -4 e 15)) (exponent-float digits e))
                      ((minusp e) (format nil "0.~A~A" (make-string (- -1 e) :initial-element #\0)
                                          digits))
                      (t (let ((whole (1+ e)))
                           (if (<= (length digits) whole)
                               (format nil "~A~A.0" digits
                                       (make-string (- whole (length digits))
                                                    :initial-element #\0))
                               (format nil "~A.~A" (subseq digits 0 whole)
                                       (subseq digits whole))))))))))

(defun exponent-float (digits e)
  (format nil "~A~:[.~A~;~*~]e~:[+~;-~]~2,'0D" (char digits 0) (= 1 (length digits))
          (subseq digits 1) (minusp e) (abs e)))

(defun write-json-string (string stream)
  (write-char #\" stream)
  (loop for char across string
        do (case char
             (#\" (write-string "\\\"" stream))
             (#\\ (write-string "\\\\" stream))
             (#\Newline (write-string "\\n" stream))
             (#\Return (write-string "\\r" stream))
             (#\Tab (write-string "\\t" stream))
             (#\Backspace (write-string "\\b" stream))
             (#\Page (write-string "\\f" stream))
             (t (if (< (char-code char) 32)
                    (format stream "\\u~(~4,'0X~)" (char-code char))
                    (write-char char stream)))))
  (write-char #\" stream))

(defun json-break (stream indent depth)
  (when indent
    (write-char #\Newline stream)
    (loop repeat (* indent depth) do (write-char #\Space stream))))

(defun write-json-sequence (stream indent depth open close items write-item)
  (write-char open stream)
  (when items
    (json-break stream indent (1+ depth))
    (loop for (item . more) on items
          do (funcall write-item item)
             (when more
               (write-string (if indent "," ", ") stream)
               (json-break stream indent (1+ depth))))
    (json-break stream indent depth))
  (write-char close stream))

(defun write-python-json (value stream indent depth)
  (flet ((write-child (child) (write-python-json child stream indent (1+ depth))))
    (typecase value
      (string (write-json-string value stream))
      (integer (format stream "~D" value))
      (float (write-string (python-float value) stream))
      (ci:json-object
       (write-json-sequence stream indent depth #\{ #\} (ci:json-object-alist value)
                            (lambda (pair)
                              (write-json-string (car pair) stream)
                              (write-string ": " stream)
                              (write-child (cdr pair)))))
      (vector (write-json-sequence stream indent depth #\[ #\] (coerce value 'list)
                                   #'write-child))
      (t (case value
           ((t) (write-string "true" stream))
           ((nil) (write-string "false" stream))
           (null (write-string "null" stream))
           (t (chat-template-fail "~S has no JSON form" value)))))))

(defun python-json (value &key indent)
  "VALUE, a JSON value as the client builds them, as Python's json.dumps writes it. INDENT
is the width of one indentation level, or NIL for a single line."
  (with-output-to-string (stream)
    (write-python-json value stream indent 0)))
