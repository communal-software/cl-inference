(in-package #:cl-inference/local/tests)

(in-suite :cl-inference/local)

(defun spm-file (&key (prefix t) (pre nil) (model "llama") (add-bos :absent)
                     (tokens (mapcar #'unescape *spm-reference-tokens*))
                     (types *spm-reference-types*) (scores *spm-reference-scores*) extra omit)
  (build-gguf
   :metadata
   (remove-if (lambda (entry) (member (first entry) omit :test #'equal))
              `(("general.architecture" 8 "llama")
                ("tokenizer.ggml.model" 8 ,model)
                ,@(when pre `(("tokenizer.ggml.pre" 8 ,pre)))
                ("tokenizer.ggml.tokens" 9 (:array 8 ,tokens))
                ("tokenizer.ggml.token_type" 9 (:array 5 ,types))
                ("tokenizer.ggml.scores" 9 (:array 6 ,scores))
                ("tokenizer.ggml.add_space_prefix" 7 ,prefix)
                ,@(unless (eq add-bos :absent) `(("tokenizer.ggml.add_bos_token" 7 ,add-bos)))
                ,@extra))))

(defmacro with-spm ((tokenizer &rest options) &body body)
  `(with-gguf-file (path (spm-file ,@options))
     (let ((,tokenizer (local:with-weights (weights path) (local:load-tokenizer weights))))
       ,@body)))

(defun hex-octets (string)
  (map '(simple-array (unsigned-byte 8) (*)) #'identity
       (loop for i from 0 below (length string) by 2
             collect (parse-integer string :start i :end (+ i 2) :radix 16))))

(defun check-spm-cases (tokenizer source)
  (loop for (kind input add special want hex) in *spm-reference-cases*
        when (eq source kind)
          do (let* ((text (unescape input))
                    (ids (local:encode tokenizer text :add-bos add :special special))
                    (expected (hex-octets hex))
                    (bytes (make-buffer))
                    (decoder (local:make-detokenizer tokenizer :special special)))
               (is (equal want (coerce ids 'list)) "~S (~S/~S/~S)" text kind add special)
               (loop for id across ids
                     when (local::token-rendered-p tokenizer id special)
                       do (put-octets bytes (local:token-bytes tokenizer id)))
               (is (equalp expected bytes) "Decoded bytes: ~S (~S/~S/~S)" text kind add special)
               (let ((decoded (babel:octets-to-string expected :encoding :utf-8 :errorp nil)))
                 (is (string= decoded (local:decode tokenizer ids :special special)))
                 (is (string= decoded
                              (with-output-to-string (out)
                                (loop for id across ids
                                      do (write-string (local:detokenizer-push decoder id) out))
                                (write-string (local:detokenizer-finish decoder) out))))))))

(test sentencepiece-pinned-synthetic-parity
  (with-spm (tokenizer) (check-spm-cases tokenizer :prefix))
  (with-spm (tokenizer :prefix nil) (check-spm-cases tokenizer :plain)))

(test sentencepiece-pinned-llama-vocabulary
  (let ((path (fixture-path "llama.cpp/models/ggml-vocab-llama-spm.gguf")))
    (if (probe-file path)
        (let ((tokenizer (local:with-weights (weights path) (local:load-tokenizer weights))))
          (check-spm-cases tokenizer :llama-vocab))
        (skip "Pinned Llama SentencePiece vocabulary is not in reference/llama.cpp/models/"))))

(test sentencepiece-selection-and-owned-vocabulary
  (dolist (pre '(nil "" "default" "ignored-for-spm"))
    (with-spm (tokenizer :pre pre)
      (is (typep tokenizer 'local:llama-spm))
      (is (not (local:tokenizer-ignore-merges-p tokenizer)))
      (is (equal '(1) (coerce (local:encode tokenizer "") 'list)))
      (is (equal '(1) (coerce (local:encode tokenizer "" :add-bos t) 'list)))
      (is (zerop (length (local:encode tokenizer "" :add-bos nil))))
      (is (typep (local:encode tokenizer "hello") '(simple-array (unsigned-byte 32) (*))))
      (is (= 1 (local:tokenizer-special-id tokenizer :bos)))
      (is (= 2 (local:tokenizer-special-id tokenizer :eos)))
      (is (= 0 (local:tokenizer-special-id tokenizer :unk)))
      (signals local:tokenizer-error (local:pre-tokenize tokenizer "hello"))
      (signals local:tokenizer-error (local:decode tokenizer #(-1)))))
  (with-spm (tokenizer :add-bos nil)
    (is (zerop (length (local:encode tokenizer "")))))
  (with-spm (tokenizer :extra '(("tokenizer.ggml.bos_token_id" 4 2)))
    (is (equal '(2) (coerce (local:encode tokenizer "") 'list)))))

(test sentencepiece-rejects-invalid-metadata
  (dolist (key '("tokens" "token_type" "scores"))
    (signals local:tokenizer-error
      (with-spm (tokenizer :omit (list (local::gguf-tokenizer-key key))) tokenizer)))
  (dolist (options (list (list :tokens '(""))
                         (list :types '(1)) (list :scores '(0.0))
                         (list :tokens (cons "<s>" (rest (mapcar #'unescape *spm-reference-tokens*))))
                         (list :types (cons 7 (rest *spm-reference-types*)))
                         (list :types (append (subseq *spm-reference-types* 0 3)
                                             '(1) (nthcdr 4 *spm-reference-types*)))
                         (list :tokens (append (subseq (mapcar #'unescape *spm-reference-tokens*) 0 3)
                                              '("<0xZZ>")
                                              (nthcdr 4 (mapcar #'unescape *spm-reference-tokens*))))))
    (signals local:tokenizer-error
      (with-gguf-file (path (apply #'spm-file options))
        (local:with-weights (weights path) (local:load-tokenizer weights)))))
  (dolist (extra '((("tokenizer.ggml.add_eos_token" 7 t))
                   (("tokenizer.ggml.remove_extra_whitespaces" 7 t))
                   (("tokenizer.ggml.bos_token_id" 4 9999))
                   (("tokenizer.ggml.add_bos_token" 8 "yes"))))
    (signals local:tokenizer-error (with-spm (tokenizer :extra extra) tokenizer)))
  (dolist (name '(float-features:single-float-positive-infinity float-features:single-float-nan))
    (signals local:tokenizer-error
      (with-spm (tokenizer :scores (cons (symbol-value name) (rest *spm-reference-scores*)))
        tokenizer)))
  (with-gguf-file (path (spm-file :model "unsupported"))
    (local:with-weights (weights path)
      (signals local:unknown-tokenizer (local:load-tokenizer weights)))))

(test sentencepiece-byte-stream-and-prefix-spaces
  (with-spm (tokenizer)
    (is (string= " hello" (local:decode tokenizer (enc tokenizer "hello")))))
  (with-spm (tokenizer :prefix nil)
    (is (string= "hello" (local:decode tokenizer (enc tokenizer "hello")))))
  (with-spm (tokenizer)
    (let ((decoder (local:make-detokenizer tokenizer)))
      (is (string= "" (local:detokenizer-push decoder (+ 3 #xc3))))
      (is (string= "é" (local:detokenizer-push decoder (+ 3 #xa9))))
      (is (string= "" (local:detokenizer-finish decoder))))
    (let ((decoder (local:make-detokenizer tokenizer)))
      (is (string= "" (local:detokenizer-push decoder (+ 3 #xf0))))
      (is (string= (string (code-char #xfffd)) (local:detokenizer-finish decoder))))))

(test sentencepiece-chat-provenance-keeps-ordinary-spans-intact
  (with-spm (tokenizer :add-bos nil)
    (flet ((segment (text kind) (local:make-chat-segment :text text :kind kind)))
      (is (equalp (enc tokenizer "hello")
                  (local::encode-chat-segments tokenizer "hello"
                                               (list (segment "hel" :markup)
                                                     (segment "lo" :text)))))
      (is (equalp (enc tokenizer "<s>")
                  (local::encode-chat-segments tokenizer "<s>"
                                               (list (segment "<" :markup)
                                                     (segment "s>" :text)))))
      (is (equalp (concatenate '(simple-array (unsigned-byte 32) (*))
                              #(1) (enc tokenizer "a<s>b"))
                  (local::encode-chat-segments tokenizer "<s>a<s>b"
                                               (list (segment "<s>" :markup)
                                                     (segment "a<s>b" :text))))))))
