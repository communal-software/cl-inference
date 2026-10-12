(in-package #:cl-inference/local/tests)

(in-suite :cl-inference/local)

;;; Expected pre-tokenizer splits come from reference/pre-splits.py, which runs llama.cpp's
;;; patterns through Python's `regex`. Strings escape ~n ~r ~t ~~ and ~u{hex}.

(defun unescape (string)
  (with-output-to-string (out)
    (loop with i = 0
          while (< i (length string))
          do (let ((char (char string i)))
               (if (char/= char #\~)
                   (progn (write-char char out) (incf i))
                   (ecase (char string (1+ i))
                     (#\n (write-char #\Newline out) (incf i 2))
                     (#\r (write-char #\Return out) (incf i 2))
                     (#\t (write-char #\Tab out) (incf i 2))
                     (#\~ (write-char #\~ out) (incf i 2))
                     (#\u (let ((close (position #\} string :start i)))
                            (write-char (code-char (parse-integer string :start (+ i 3) :end close
                                                                         :radix 16))
                                        out)
                            (setf i (1+ close))))))))))

(defparameter *llama3-splits*
  '(("Hello world" "Hello" " world")
    ("It's 'Tis I'LL we're THEY'VE I'M she'd" "It" "'s" " '" "Tis" " I" "'LL" " we" "'re" " THEY" "'VE" " I" "'M" " she" "'d")
    ("1234567" "123" "456" "7")
    ("a1b22c333d4444" "a" "1" "b" "22" "c" "333" "d" "444" "4")
    ("Hello, world!!~n~nNext" "Hello" "," " world" "!!~n~n" "Next")
    ("  leading and trailing  " " " " leading" " and" " trailing" "  ")
    ("a  b   c" "a" " " " b" "  " " c")
    ("line1~r~nline2~r~n~r~nline3  ~n  x" "line" "1" "~r~n" "line" "2" "~r~n~r~n" "line" "3" "  ~n" " " " x")
    ("tab~there~t~tthere" "tab" "~there" "~t" "~tthere")
    ("h~u{e9}llo w~u{f6}rld" "h~u{e9}llo" " w~u{f6}rld")
    ("~u{65e5}~u{672c}~u{8a9e}~u{306e}~u{30c6}~u{30ad}~u{30b9}~u{30c8} ~u{4e2d}~u{6587}" "~u{65e5}~u{672c}~u{8a9e}~u{306e}~u{30c6}~u{30ad}~u{30b9}~u{30c8}" " ~u{4e2d}~u{6587}")
    ("x~u{1f600}y ~u{1f600}~u{1f600} !~u{1f600}" "x" "~u{1f600}y" " ~u{1f600}~u{1f600}" " !~u{1f600}")
    ("e~u{301}cole" "e" "~u{301}cole")
    ("price: $12.50!" "price" ":" " $" "12" "." "50" "!")
    (" 12" " " "12")
    ("   " "   ")
    ("~n" "~n")
    (" ~n " " ~n" " ")
    ("...ok?!~r~nfoo" "..." "ok" "?!~r~n" "foo")
    ("don't stop" "don" "'t" " stop")
    ("'s" "'s")
    ("x'" "x" "'")
    ("'" "'")
    ("foo~u{a0}bar" "foo" "~u{a0}bar")
    ("a~u{2003}b" "a" "~u{2003}b")
    ("a~u{3000}b" "a" "~u{3000}b")
    ("C++/Java_script-ish" "C" "++/" "Java" "_script" "-ish")
    ("snake_case_name" "snake" "_case" "_name")
    ("a~n ~nb" "a" "~n ~n" "b")
    ("text   " "text" "   ")
    ("  ~n  " "  ~n" "  ")
    ("~u{663}~u{664} ~u{b2}~u{b3}" "~u{663}~u{664}" " " "~u{b2}~u{b3}")
    ("a~u{2028}b" "a" "~u{2028}b")))

(defparameter *qwen2-splits*
  '(("Hello world" "Hello" " world")
    ("It's 'Tis I'LL we're THEY'VE I'M she'd" "It" "'s" " '" "Tis" " I" "'LL" " we" "'re" " THEY" "'VE" " I" "'M" " she" "'d")
    ("1234567" "1" "2" "3" "4" "5" "6" "7")
    ("a1b22c333d4444" "a" "1" "b" "2" "2" "c" "3" "3" "3" "d" "4" "4" "4" "4")
    ("Hello, world!!~n~nNext" "Hello" "," " world" "!!~n~n" "Next")
    ("  leading and trailing  " " " " leading" " and" " trailing" "  ")
    ("a  b   c" "a" " " " b" "  " " c")
    ("line1~r~nline2~r~n~r~nline3  ~n  x" "line" "1" "~r~n" "line" "2" "~r~n~r~n" "line" "3" "  ~n" " " " x")
    ("tab~there~t~tthere" "tab" "~there" "~t" "~tthere")
    ("h~u{e9}llo w~u{f6}rld" "h~u{e9}llo" " w~u{f6}rld")
    ("~u{65e5}~u{672c}~u{8a9e}~u{306e}~u{30c6}~u{30ad}~u{30b9}~u{30c8} ~u{4e2d}~u{6587}" "~u{65e5}~u{672c}~u{8a9e}~u{306e}~u{30c6}~u{30ad}~u{30b9}~u{30c8}" " ~u{4e2d}~u{6587}")
    ("x~u{1f600}y ~u{1f600}~u{1f600} !~u{1f600}" "x" "~u{1f600}y" " ~u{1f600}~u{1f600}" " !~u{1f600}")
    ("e~u{301}cole" "e" "~u{301}cole")
    ("price: $12.50!" "price" ":" " $" "1" "2" "." "5" "0" "!")
    (" 12" " " "1" "2")
    ("   " "   ")
    ("~n" "~n")
    (" ~n " " ~n" " ")
    ("...ok?!~r~nfoo" "..." "ok" "?!~r~n" "foo")
    ("don't stop" "don" "'t" " stop")
    ("'s" "'s")
    ("x'" "x" "'")
    ("'" "'")
    ("foo~u{a0}bar" "foo" "~u{a0}bar")
    ("a~u{2003}b" "a" "~u{2003}b")
    ("a~u{3000}b" "a" "~u{3000}b")
    ("C++/Java_script-ish" "C" "++/" "Java" "_script" "-ish")
    ("snake_case_name" "snake" "_case" "_name")
    ("a~n ~nb" "a" "~n ~n" "b")
    ("text   " "text" "   ")
    ("  ~n  " "  ~n" "  ")
    ("~u{663}~u{664} ~u{b2}~u{b3}" "~u{663}" "~u{664}" " " "~u{b2}" "~u{b3}")
    ("a~u{2028}b" "a" "~u{2028}b")))


;;; Fixture vocabulary: ids 0-255 are the bytes, then merged tokens and three specials.

(defun byte-string (&rest bytes)
  (map 'string (lambda (byte) (aref local::*byte-chars* byte)) bytes))

(defun fixture-tokens ()
  (append (loop for byte below 256 collect (byte-string byte))
          (list "he" "ll" "hell" "hello" "ab" "bc" "aa" (byte-string #xC3 #xA9)
                "<s>" "</s>" "<think>")))

(defun fixture-types ()
  (append (make-list 264 :initial-element 1) '(3 3 4)))

(defun fixture-merges ()
  (list "h e" "l l" "he ll" "hell o" "b c" "a b" "a a"
        (format nil "~A ~A" (byte-string #xC3) (byte-string #xA9))))

(defun tokenizer-file (&key (model "gpt2") (pre "llama-bpe") (tokens (fixture-tokens))
                         (types (fixture-types)) (merges (fixture-merges)) (add-bos t)
                         (specials '(("bos_token_id" 264) ("eos_token_id" 265))) omit)
  (build-gguf
   :metadata (remove-if
              (lambda (entry) (member (first entry) omit :test #'string=))
              `(("general.architecture" 8 "llama")
                ,@(when model `(("tokenizer.ggml.model" 8 ,model)))
                ,@(when pre `(("tokenizer.ggml.pre" 8 ,pre)))
                ("tokenizer.ggml.tokens" 9 (:array 8 ,tokens))
                ("tokenizer.ggml.token_type" 9 (:array 5 ,types))
                ("tokenizer.ggml.merges" 9 (:array 8 ,merges))
                ("tokenizer.ggml.add_bos_token" 7 ,add-bos)
                ,@(loop for (name id) in specials
                        collect `(,(format nil "tokenizer.ggml.~A" name) 4 ,id))))))

(defmacro with-tokenizer ((var &rest file-options) &body body)
  (let ((path (gensym "PATH")))
    `(with-gguf-file (,path (tokenizer-file ,@file-options))
       (let ((,var (local:with-weights (weights ,path) (local:load-tokenizer weights))))
         ,@body))))

(defun ids (&rest ids)
  (coerce ids '(simple-array (unsigned-byte 32) (*))))

(defun enc (tokenizer text &rest options)
  (apply #'local:encode tokenizer text :add-bos nil options))

;;; Definitions used by the tests below

(local:define-tokenizer test-no-prefix-bpe (local:llama-bpe)
  (:pre-tokenizer (letters))
  (:format :gguf (:pre "test-no-prefix")))

(local:define-tokenizer test-typo-bpe (local:llama-bpe)
  (:pre-tokenizer (digits :maxx 2))
  (:format :gguf (:pre "test-typo")))

(local:define-tokenizer test-unknown-rule-bpe (local:llama-bpe)
  (:pre-tokenizer (no-such-rule))
  (:format :gguf (:pre "test-unknown-rule")))

(local:define-tokenizer test-empty-bpe (local:byte-level-bpe)
  (:format :gguf (:model "gpt2") (:pre "test-empty")))

(local:define-tokenizer test-dup-a (local:llama-bpe)
  (:format :gguf (:pre "test-dup")))

(local:define-tokenizer test-dup-b (local:llama-bpe)
  (:format :gguf (:pre "test-dup")))

(defmethod local:rule-matcher ((kind (eql :test-hash)) options)
  (declare (ignore options))
  (lambda (text start) (and (char= (schar text start) #\#) (1+ start))))

(local:define-tokenizer test-hash-bpe (local:llama-bpe)
  (:pre-tokenizer (test-hash))
  (:format :gguf (:pre "test-hash")))

;;; The DSL

(test tokenizer-selected-by-pre
  (dolist (case '(("llama-bpe" local:llama-bpe) ("llama3" local:llama-bpe)
                  ("falcon3" local:llama-bpe) ("qwen2" local:qwen2)
                  ("deepseek-r1-qwen" local:qwen2)))
    (with-tokenizer (tokenizer :pre (first case))
      (is (eq (second case) (type-of tokenizer)) "pre ~S" (first case))
      (is (eq :gguf (local:tokenizer-format tokenizer))))))

(test tokenizer-unknown-pre-or-model
  (dolist (options '((:pre "no-such-pre") (:pre nil) (:model "llama") (:model nil)))
    (with-gguf-file (path (apply #'tokenizer-file options))
      (signals local:unknown-tokenizer
        (local:with-weights (w path) (local:load-tokenizer w))))))

(test tokenizer-unknown-carries-names
  (with-gguf-file (path (tokenizer-file :pre "no-such-pre"))
    (handler-case (local:with-weights (w path) (local:load-tokenizer w))
      (local:unknown-tokenizer (condition)
        (is (equal "gpt2" (local:unknown-tokenizer-model condition)))
        (is (equal "no-such-pre" (local:unknown-tokenizer-pre condition)))))))

(test tokenizer-ambiguous-match
  (with-gguf-file (path (tokenizer-file :pre "test-dup"))
    (signals local:tokenizer-error
      (local:with-weights (w path) (local:load-tokenizer w)))))

(test tokenizer-child-overrides-rules-in-place
  (is (equal '((:contractions :case-insensitive t) (:letters :prefix :non-letter-non-newline)
               (:digits :max 3) (:punctuation :leading-space t :trailing-newlines t)
               (:newlines) (:whitespace))
             (local:tokenizer-pre-rules 'local:llama-bpe)))
  (is (equal '(:digits :max 1) (assoc :digits (local:tokenizer-pre-rules 'local:qwen2))))
  (is (equal '(:contractions :letters :digits :punctuation :newlines :whitespace)
             (mapcar #'car (local:tokenizer-pre-rules 'local:qwen2))))
  (is (equal "gpt2" (local:tokenizer-format-option 'local:qwen2 :gguf :model))))

(test tokenizer-child-adds-rule-at-the-end
  (is (equal '(:contractions :letters :digits :punctuation :newlines :whitespace :test-hash)
             (mapcar #'car (local:tokenizer-pre-rules 'test-hash-bpe))))
  (with-tokenizer (tokenizer :pre "test-hash")
    (is (eq 'test-hash-bpe (type-of tokenizer)))))

(test tokenizer-redefinition-propagates-to-children
  (local:define-tokenizer prop-parent (local:byte-level-bpe) (:pre-tokenizer (digits :max 2)))
  (local:define-tokenizer prop-child (prop-parent) (:format :gguf (:pre "prop")))
  (is (equal '((:digits :max 2)) (local:tokenizer-pre-rules 'prop-child)))
  (local:define-tokenizer prop-parent (local:byte-level-bpe) (:pre-tokenizer (digits :max 5)))
  (is (equal '((:digits :max 5)) (local:tokenizer-pre-rules 'prop-child))))

(test tokenizer-rejects-malformed-definitions
  (dolist (form '((local:define-tokenizer bad () (:nonsense))
                  (local:define-tokenizer bad () nonsense)
                  (local:define-tokenizer bad () (:pre-tokenizer 5))
                  (local:define-tokenizer bad () (:pre-tokenizer (digits :max)))
                  (local:define-tokenizer bad () (:pre-tokenizer (digits 3 4)))
                  (local:define-tokenizer bad () (:format "gguf"))
                  (local:define-tokenizer bad () (:format :gguf (:model 5)))
                  (local:define-tokenizer bad () (:format :gguf (:model "a" "b")))
                  (local:define-tokenizer bad () (:format :gguf (:pre)))
                  (local:define-tokenizer bad () (:format :gguf (:pre 5)))
                  (local:define-tokenizer bad () (:format :gguf nonsense))))
    (signals local:tokenizer-error (macroexpand-1 form))))

(test tokenizer-rejects-bad-rules-at-load
  (dolist (pre '("test-typo" "test-unknown-rule" "test-empty"))
    (with-gguf-file (path (tokenizer-file :pre pre))
      (signals local:tokenizer-error
        (local:with-weights (w path) (local:load-tokenizer w))))))

;;; Pre-tokenization

(defun check-splits (pre table)
  (with-tokenizer (tokenizer :pre pre)
    (dolist (row table)
      (let ((text (unescape (first row))))
        (is (equal (mapcar #'unescape (rest row)) (local:pre-tokenize tokenizer text))
            "~A split of ~S" pre text)))))

(test pre-tokenize-llama3-matches-reference
  (check-splits "llama-bpe" *llama3-splits*))

(test pre-tokenize-qwen2-matches-reference
  (check-splits "qwen2" *qwen2-splits*))

(test pre-tokenize-empty-text
  (with-tokenizer (tokenizer)
    (is (null (local:pre-tokenize tokenizer "")))))

(test pre-tokenize-rule-override
  (with-tokenizer (tokenizer :pre "test-no-prefix")
    (is (equal '("hello" " " "world") (local:pre-tokenize tokenizer "hello world")))))

(test pre-tokenize-custom-rule-kind
  (is (functionp (local:rule-matcher :test-hash nil)))
  (is (= 1 (funcall (local:rule-matcher :test-hash nil) (coerce "#a" '(simple-array character (*))) 0))))

;;; BPE

(test encode-merges-by-rank
  (with-tokenizer (tokenizer)
    (is (equalp (ids 259) (enc tokenizer "hello")))
    (is (equalp (ids 97 261) (enc tokenizer "abc")))
    (is (equalp (ids 259 32 119 111 114 108 100) (enc tokenizer "hello world")))
    (is (equalp (ids 104 263 257 111) (enc tokenizer (unescape "h~u{e9}llo"))))
    (is (equalp (ids) (enc tokenizer "")))))

(test encode-ties-merge-leftmost-first
  (with-tokenizer (tokenizer)
    (is (equalp (ids 262 97) (enc tokenizer "aaa")))
    (is (equalp (ids 262 262) (enc tokenizer "aaaa")))))

(test encode-long-piece
  (with-tokenizer (tokenizer)
    (let ((ids (enc tokenizer (make-string 5000 :initial-element #\a))))
      (is (= 2500 (length ids)))
      (is (every (lambda (id) (= id 262)) ids)))))

(test encode-round-trips
  (with-tokenizer (tokenizer)
    (dolist (text (list "hello world" (unescape "h~u{e9}llo w~u{f6}rld") (unescape "x~u{1f600}y")
                        (unescape "~u{65e5}~u{672c}~u{8a9e} text") (unescape "a~r~n  b~t")
                        "1234567 and 'quoted' text!" ""))
      (is (equal text (local:decode tokenizer (enc tokenizer text))) "round trip of ~S" text))))

(test encode-type-checked
  (with-tokenizer (tokenizer)
    (signals type-error (local:encode tokenizer 42))))

(test encode-add-bos
  (with-tokenizer (tokenizer)
    (is (equalp (ids 264 259) (local:encode tokenizer "hello")))
    (is (equalp (ids 259) (local:encode tokenizer "hello" :add-bos nil))))
  (with-tokenizer (tokenizer :add-bos nil)
    (is (equalp (ids 259) (local:encode tokenizer "hello")))
    (is (equalp (ids 264 259) (local:encode tokenizer "hello" :add-bos t))))
  (with-tokenizer (tokenizer :add-bos nil :specials nil)
    (signals local:tokenizer-error (local:encode tokenizer "hello" :add-bos t))))

(test encode-special-tokens
  (with-tokenizer (tokenizer)
    (is (equalp (ids 264 259 265) (enc tokenizer "<s>hello</s>" :special t)))
    (is (equalp (ids 256 264 257 111) (enc tokenizer "he<s>llo" :special t)))
    (let ((plain (enc tokenizer "<s>hello")))
      (is (not (find 264 plain)))
      (is (equal "<s>hello" (local:decode tokenizer plain))))))

(test user-defined-tokens-match-without-special
  (with-tokenizer (tokenizer)
    (is (find 266 (enc tokenizer "a<think>b")))
    (is (equal "a<think>b" (local:decode tokenizer (ids 97 266 98))))))

(test decode-leaves-out-control-tokens
  (with-tokenizer (tokenizer)
    (is (equal "hello" (local:decode tokenizer (ids 264 259 265))))
    (is (equal "<s>hello</s>" (local:decode tokenizer (ids 264 259 265) :special t)))))

(test decode-rejects-invalid-ids
  (with-tokenizer (tokenizer)
    (dolist (id (list 267 99999 -1 "x" nil))
      (signals local:tokenizer-error (local:decode tokenizer (list id))))))

(test decode-replaces-invalid-utf-8
  (with-tokenizer (tokenizer)
    (is (equal (string (code-char #xFFFD)) (local:decode tokenizer (ids #xFF))))
    (is (equal (format nil "a~Ab" (code-char #xFFFD)) (local:decode tokenizer (ids 97 #xC3 98))))))

(test vocabulary-accessors
  (with-tokenizer (tokenizer)
    (is (= 267 (local:tokenizer-vocab-size tokenizer)))
    (is (= 259 (local:token-id tokenizer "hello")))
    (is (null (local:token-id tokenizer "nope")))
    (is (equalp #(104 101 108 108 111) (local:token-bytes tokenizer 259)))
    (is (equalp #(#xC3 #xA9) (local:token-bytes tokenizer 263)))
    (is (equalp (coerce "<s>" 'list) (map 'list #'code-char (local:token-bytes tokenizer 264))))
    (is (= 264 (local:tokenizer-special-id tokenizer :bos)))
    (is (= 265 (local:tokenizer-special-id tokenizer :eos)))
    (is (null (local:tokenizer-special-id tokenizer :eot)))
    (is (local:tokenizer-add-bos-p tokenizer))
    (is (local:tokenizer-special-p tokenizer 264))
    (is (local:tokenizer-special-p tokenizer 266))
    (is (not (local:tokenizer-special-p tokenizer 259)))
    (signals local:tokenizer-error (local:token-bytes tokenizer 267))))

(test llama-bpe-emits-whole-pieces-found-in-the-vocabulary
  (let ((merges '("h e" "l l" "he ll")))   ; no merge reaches "hello"
    (with-tokenizer (tokenizer :pre "llama-bpe" :merges merges)
      (is (local:tokenizer-ignore-merges-p tokenizer))
      (is (equalp (ids 259) (enc tokenizer "hello")))
      (is (equalp (ids 259 32 119 111 114 108 100) (enc tokenizer "hello world"))))
    (with-tokenizer (tokenizer :pre "qwen2" :merges merges)
      (is (not (local:tokenizer-ignore-merges-p tokenizer)))
      (is (equalp (ids 258 111) (enc tokenizer "hello"))))))

(test add-bos-defaults-follow-the-tokenizer
  (let ((omit '("tokenizer.ggml.add_bos_token")))
    (with-tokenizer (tokenizer :pre "llama-bpe" :omit omit)
      (is (local:tokenizer-add-bos-p tokenizer))
      (is (equalp (ids 264 259) (local:encode tokenizer "hello"))))
    (with-tokenizer (tokenizer :pre "qwen2" :omit omit)
      (is (not (local:tokenizer-add-bos-p tokenizer))))
    (with-tokenizer (tokenizer :pre "llama-bpe" :add-bos nil)
      (is (not (local:tokenizer-add-bos-p tokenizer))))))

;;; Streaming decode

(defun stream-decode (detokenizer ids)
  (with-output-to-string (out)
    (map nil (lambda (id) (write-string (local:detokenizer-push detokenizer id) out)) ids)
    (write-string (local:detokenizer-finish detokenizer) out)))

(test detokenizer-holds-back-partial-characters
  (with-tokenizer (tokenizer)
    (let ((d (local:make-detokenizer tokenizer)))
      (is (equal "a" (local:detokenizer-push d 97)))
      (is (equal "" (local:detokenizer-push d #xC3)))
      (is (equal (unescape "~u{e9}") (local:detokenizer-push d #xA9)))
      (is (equal "" (local:detokenizer-push d #xF0)))
      (is (equal "" (local:detokenizer-push d #x9F)))
      (is (equal "" (local:detokenizer-push d #x98)))
      (is (equal (unescape "~u{1f600}") (local:detokenizer-push d #x80)))
      (is (equal "" (local:detokenizer-finish d))))))

(test detokenizer-finish-flushes-dangling-bytes
  (with-tokenizer (tokenizer)
    (let ((d (local:make-detokenizer tokenizer)))
      (local:detokenizer-push d #xC3)
      (is (equal (string (code-char #xFFFD)) (local:detokenizer-finish d)))
      (is (equal "a" (local:detokenizer-push d 97))))))

(test detokenizer-follows-the-special-flag
  (with-tokenizer (tokenizer)
    (is (equal "" (local:detokenizer-push (local:make-detokenizer tokenizer) 264)))
    (is (equal "<s>" (local:detokenizer-push (local:make-detokenizer tokenizer :special t) 264)))
    (is (equal "<think>" (local:detokenizer-push (local:make-detokenizer tokenizer) 266)))))

(test detokenizer-matches-decode
  (with-tokenizer (tokenizer)
    (let* ((text (unescape "h~u{e9}llo w~u{f6}rld ~u{1f600} hello ~u{65e5}~u{672c}"))
           (ids (enc tokenizer text)))
      (is (equal (local:decode tokenizer ids)
                 (stream-decode (local:make-detokenizer tokenizer) ids)))
      (is (equal text (stream-decode (local:make-detokenizer tokenizer) ids))))))

(test detokenizer-rejects-invalid-ids
  (with-tokenizer (tokenizer)
    (signals local:tokenizer-error (local:detokenizer-push (local:make-detokenizer tokenizer) 99999))))

;;; Malformed metadata

(test tokenizer-rejects-malformed-metadata
  (dolist (options
           `((:types (1 1 1))
             (:merges ("hello"))
             (:merges ("h e" "a b c"))
             (:merges ("h  e"))
             (:merges (" h"))
             (:merges ("h zzz"))
             (:merges ("h l"))
             (:specials (("bos_token_id" 9999)))
             (:specials (("eos_token_id" -1)))
             (:tokens ,(cdr (fixture-tokens)) :types ,(cdr (fixture-types)))
             (:tokens ,(append (fixture-tokens) (list (string (code-char #x4e2d))))
              :types ,(append (fixture-types) '(1)))
             (:omit ("tokenizer.ggml.tokens"))
             (:omit ("tokenizer.ggml.merges"))))
    (with-gguf-file (path (apply #'tokenizer-file options))
      (signals local:tokenizer-error
        (local:with-weights (w path) (local:load-tokenizer w))
        "options ~S" options))))

(test tokenizer-loads-without-token-types
  (with-tokenizer (tokenizer :omit '("tokenizer.ggml.token_type"))
    (is (equalp (ids 259) (enc tokenizer "hello")))
    (is (not (local:tokenizer-special-p tokenizer 264)))))
