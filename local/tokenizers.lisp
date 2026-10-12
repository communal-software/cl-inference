(in-package #:cl-inference/local)

;;; Pre-tokenizer rules and `pre` aliases follow llama.cpp's llama-vocab.cpp.

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
