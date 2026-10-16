(in-package #:cl-inference/local/tests)

(in-suite :cl-inference/local)

(defmacro with-qwen35 ((tokenizer &rest options) &body body)
  `(with-tokenizer (,tokenizer :pre "qwen35"
                    :tokens (mapcar #'unescape *qwen35-reference-tokens*)
                    :types *qwen35-reference-types*
                    :merges (mapcar #'unescape *qwen35-reference-merges*)
                    :specials '(("bos_token_id" 256) ("eos_token_id" 257))
                    ,@options)
     ,@body))

(defun check-qwen35-cases (tokenizer source)
  (loop for (kind input add special want hex) in *qwen35-reference-cases*
        when (eq kind source)
          do (let* ((text (unescape input))
                    (ids (local:encode tokenizer text :add-bos add :special special))
                    (expected (map '(vector (unsigned-byte 8)) #'identity
                                   (loop for i below (length hex) by 2
                                         collect (parse-integer hex :start i :end (+ i 2) :radix 16))))
                    (bytes (make-buffer))
                    (decoder (local:make-detokenizer tokenizer :special special)))
               (is (equal want (coerce ids 'list)) "~S (~S/~S/~S)" text source add special)
               (loop for id across ids
                     when (local::token-rendered-p tokenizer id special)
                       do (put-octets bytes (local:token-bytes tokenizer id)))
               (is (equalp expected bytes) "Decoded bytes: ~S (~S/~S/~S)" text source add special)
               (let ((decoded (babel:octets-to-string expected :encoding :utf-8 :errorp nil)))
                 (is (string= decoded (local:decode tokenizer ids :special special)))
                 (is (string= decoded
                              (with-output-to-string (out)
                                (loop for id across ids
                                      do (write-string (local:detokenizer-push decoder id) out))
                                (write-string (local:detokenizer-finish decoder) out))))))))

(test qwen35-pinned-offline-parity
  (with-qwen35 (tokenizer :omit '("tokenizer.ggml.add_bos_token"))
    (is (eq 'local:qwen35 (type-of tokenizer)))
    (dolist (row *qwen35-reference-splits*)
      (destructuring-bind (input pieces) row
        (is (equal (mapcar #'unescape pieces) (local:pre-tokenize tokenizer (unescape input)))
            "Qwen3.5 splits ~S" input)))
    (check-qwen35-cases tokenizer :synthetic)))

(test qwen35-pinned-real-vocabulary-parity
  (let ((path (fixture-path "llama.cpp/models/ggml-vocab-qwen35.gguf")))
    (if (probe-file path)
        (local:with-weights (weights path)
          (let ((tokenizer (local:load-tokenizer weights)))
            (is (eq 'local:qwen35 (type-of tokenizer)))
            (check-qwen35-cases tokenizer :vocab)))
        (skip "Pinned Qwen3.5 vocabulary is not in reference/llama.cpp/models/"))))

(test qwen35-defaults-and-overrides
  (with-qwen35 (tokenizer :omit '("tokenizer.ggml.add_bos_token"))
    (is (not (local:tokenizer-add-bos-p tokenizer)))
    (is (not (local:tokenizer-ignore-merges-p tokenizer)))
    (is (equalp (ids) (local:encode tokenizer "")))
    (is (equalp (ids 256) (local:encode tokenizer "" :add-bos t))))
  (with-qwen35 (tokenizer :add-bos t)
    (is (local:tokenizer-add-bos-p tokenizer))
    (is (= 256 (aref (local:encode tokenizer "hello") 0)))
    (is (/= 256 (aref (local:encode tokenizer "hello" :add-bos nil) 0)))))

(test qwen35-rule-order-and-mark-options
  (is (equal '(:contractions :letters :digits :punctuation :newlines :whitespace)
             (mapcar #'car (local:tokenizer-pre-rules 'local:qwen35))))
  (dolist (input '("~u{301}" "~u{903}" "~u{20dd}" "~u{1d165}"))
    (let* ((text (unescape input))
           (letters (local:rule-matcher :letters '(:include-marks t)))
           (plain (local:rule-matcher :letters nil))
           (punctuation (local:rule-matcher :punctuation '(:exclude-marks t))))
      (is (= 1 (funcall letters text 0)))
      (is (null (funcall plain text 0)))
      (is (null (funcall punctuation text 0)))))
  (with-qwen35 (tokenizer :add-bos nil)
    (let ((text (unescape "e~u{301}")))
      (is (equal (list text) (local:pre-tokenize tokenizer text)))
      (is (= 1 (length (local:encode tokenizer text))))
      (is (equal text (local:decode tokenizer (local:encode tokenizer text))))))
  (with-tokenizer (tokenizer :pre "qwen2")
    (is (equal (list "e" (unescape "~u{301}"))
               (local:pre-tokenize tokenizer (unescape "e~u{301}"))))))

(test qwen35-normal-end-marker-is-control
  (dolist (pre '("qwen35" "qwen2" "llama-bpe" "default"))
    (with-tokenizer (tokenizer :pre pre :add-bos nil
                              :types (append (make-list 264 :initial-element 1) '(3 1 4)))
      (if (equal pre "qwen35")
          (progn
            (is (local:tokenizer-special-p tokenizer 265))
            (is (equalp (ids 265) (enc tokenizer "</s>" :special t)))
            (is (not (find 265 (enc tokenizer "</s>"))))
            (is (equal "" (local:decode tokenizer (ids 265))))
            (is (equal "</s>" (local:decode tokenizer (ids 265) :special t))))
          (is (not (local:tokenizer-special-p tokenizer 265)))))))
