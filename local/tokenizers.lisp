(in-package #:cl-inference/local)

;;; Pre-tokenizer rules and `pre` aliases follow llama.cpp's llama-vocab.cpp.

(define-tokenizer default-bpe (byte-level-bpe)
  (:pre-tokenizer-stages
   (symbols (punctuation-symbols))
   (words (contractions)
          (letters :prefix :space)
          (digits :prefix :space :max nil)
          (punctuation :leading-space t)
          (whitespace))
   (numbers (digits :max nil))
   (triples (digits :min 3 :max 3 :ascii-only t)))
  (:format :gguf (:model "gpt2") (:pre "default") (:ignore-merges nil) (:add-bos nil)))

(define-tokenizer llama-bpe (byte-level-bpe)
  (:pre-tokenizer
   (contractions :case-insensitive t)
   (letters :prefix :non-letter-non-newline)
   (digits :max 3)
   (punctuation :leading-space t :trailing-newlines t)
   (newlines)
   (whitespace))
  (:format :gguf
           (:model "gpt2")
           (:pre "llama-bpe" "llama3" "llama-v3" "falcon3" "falcon-h1" "pixtral" "midm-2.0"
                 "lfm2")
           (:ignore-merges t)
           (:add-bos t)))

(define-tokenizer qwen2 (llama-bpe)
  (:pre-tokenizer (digits :max 1))
  (:format :gguf (:pre "qwen2" "deepseek-r1-qwen" "kormo") (:ignore-merges nil) (:add-bos nil)))

(define-tokenizer qwen35 (qwen2)
  (:pre-tokenizer
   (letters :prefix :non-letter-non-newline :include-marks t)
   (punctuation :leading-space t :trailing-newlines t :exclude-marks t))
  (:format :gguf (:pre "qwen35")))

(defmethod build-vocabulary :before ((tokenizer qwen35) tokens types)
  (when (eq (tokenizer-format tokenizer) :gguf)
    ;; The pinned llama.cpp loader treats even a normal </s> entry as control.
    (let ((id (position "</s>" tokens :test #'equal)))
      (when id (setf (aref types id) +token-control+)))))
