(in-package #:cl-inference/local/tests)
(in-suite :cl-inference/local)

(defun hf-tokenizer-fixture (arch)
  (let ((vocab (ci::json-object)) (added nil)
        (tokens (append (fixture-tokens)
                        '("<|im_start|>" "<|im_end|>" "</think>" "<|python_tag|>"
                          "<|start_header_id|>" "<|end_header_id|>" "<|eot_id|>"
                          "<|eom_id|>" "<|end_of_text|>"))))
    (loop for token in tokens for id from 0
          do (setf (ci::json-get vocab token) id)
             (when (>= id 264)
               (push (ci::json-object "id" id "content" token "special" t
                                     "single_word" nil "lstrip" nil "rstrip" nil "normalized" nil) added)))
    (ci::json-object
     "model" (ci::json-object "type" "BPE" "vocab" vocab
                              "merges" (coerce (fixture-merges) 'vector)
                              "ignore_merges" (equal arch "llama"))
     "normalizer" (if (equal arch "qwen3") (ci::json-object "type" "NFC") 'cl:null)
     "pre_tokenizer" (ci::json-object
                      "type" "Sequence" "pretokenizers"
                      (vector (ci::json-object "type" "Split" "pattern"
                                               (ci::json-object "Regex" (if (equal arch "llama") st::*llama-pattern* st::*qwen-pattern*))
                                               "behavior" "Isolated" "invert" nil)
                              (ci::json-object "type" "ByteLevel" "add_prefix_space" nil "use_regex" nil)))
     "post_processor" 'cl:null "decoder" (ci::json-object "type" "ByteLevel")
     "added_tokens" (coerce (nreverse added) 'vector))))

(defun write-hf-tokenizer-fixture (directory arch)
  (write-hf-json directory "tokenizer.json" (hf-tokenizer-fixture arch))
  (write-hf-json directory "tokenizer_config.json"
                 (ci::json-object "bos_token" "<s>" "eos_token" "<|im_end|>"
                                  "add_bos_token" nil "chat_template"
                                  (if (equal arch "llama") "<|start_header_id|> Cutting Knowledge Date"
                                      "<|im_start|> multi_step_tool"))))

(test hf-tokenizer-bpe-specials-normalization-and-template
  (dolist (arch '("llama" "qwen3"))
    (with-hf-directory (directory)
      (write-hf-model directory arch (spec* :qk-norm (equal arch "qwen3"))
                      (weight-table (spec* :qk-norm (equal arch "qwen3"))))
      (write-hf-tokenizer-fixture directory arch)
      (local:with-weights (w directory)
        (let ((tokenizer (local:load-tokenizer w)) (template (local:load-chat-template w)))
          (is (equalp #(259) (local:encode tokenizer "hello")))
          (is (= 268 (local:tokenizer-special-id tokenizer :eos)))
          (is (equalp #(267) (local:encode tokenizer "<|im_start|>" :special t)))
          (is (equal "hello é" (local:decode tokenizer (local:encode tokenizer "hello é"))))
          (when (equal arch "qwen3")
            (is (equalp (local:encode tokenizer "é")
                        (local:encode tokenizer (format nil "e~A" (code-char #x301))))))
          (is (typep template (if (equal arch "llama") 'local:llama3-chat 'local:qwen3-chat)))
          (is (plusp (length (local:encode-chat template tokenizer '((:role :user :content "hi")))))))))))

(test hf-bpe-cleanup-configuration
  (dolist (arch '("llama" "qwen3"))
    (with-hf-directory (directory)
      (write-hf-model directory arch (spec*) (weight-table (spec*)))
      (write-hf-tokenizer-fixture directory arch)
      (let ((config (ci::json-object "clean_up_tokenization_spaces" t))
            (text "hello ,  world !"))
        (write-hf-json directory "tokenizer_config.json" config)
        (local:with-weights (w directory)
          (let ((tokenizer (local:load-tokenizer w)))
            (is (equal text (local:decode tokenizer (local:encode tokenizer text))))))
        (setf (ci::json-get config "clean_up_tokenization_spaces_for_bpe_even_though_it_will_corrupt_output") t)
        (write-hf-json directory "tokenizer_config.json" config)
        (local:with-weights (w directory)
          (signals local:tokenizer-error (local:load-tokenizer w)))))))

(test hf-tokenizer-rejects-unsupported-files
  (with-hf-directory (directory)
    (write-hf-model directory "qwen3" (spec*) (weight-table (spec*)))
    (local:with-weights (w directory) (signals local:tokenizer-error (local:load-tokenizer w)))
    (write-hf-tokenizer-fixture directory "qwen3")
    (dolist (key '("normalizer" "pre_tokenizer" "decoder"))
      (let ((object (hf-tokenizer-fixture "qwen3")))
        (setf (ci::json-get object key) (ci::json-object "type" "unsupported"))
        (write-hf-json directory "tokenizer.json" object)
        (local:with-weights (w directory) (signals local:tokenizer-error (local:load-tokenizer w)))))
    (write-hf-tokenizer-fixture directory "qwen3")
    (write-hf-json directory "tokenizer_config.json" (ci::json-object "chat_template" "{{ unknown }}"))
    (local:with-weights (w directory) (signals local:chat-template-error (local:load-chat-template w)))))

(test hf-llama-bos-postprocessor
  (with-hf-directory (directory)
    (write-hf-model directory "llama" (spec*) (weight-table (spec*)))
    (let ((object (hf-tokenizer-fixture "llama")))
      (setf (ci::json-get object "post_processor")
            (ci::json-object "type" "Sequence" "processors"
             (vector (ci::json-object "type" "ByteLevel")
                     (ci::json-object "type" "TemplateProcessing"
                                      "single" (vector (ci::json-object "SpecialToken" (ci::json-object "id" "<s>" "type_id" 0))
                                                       (ci::json-object "Sequence" (ci::json-object "id" "A" "type_id" 0)))
                                      "special_tokens" (ci::json-object "<s>" (ci::json-object "ids" #(264) "tokens" #("<s>")))))))
      (write-hf-json directory "tokenizer.json" object)
      (write-hf-json directory "tokenizer_config.json" (ci::json-object "bos_token" "<s>" "add_bos_token" t))
      (local:with-weights (w directory)
        (is (equalp #(264 259) (local:encode (local:load-tokenizer w) "hello"))))
      (write-hf-json directory "tokenizer_config.json" (ci::json-object "bos_token" "</s>"))
      (local:with-weights (w directory) (signals local:tokenizer-error (local:load-tokenizer w))))))

(test hf-tokenizer-malformed-nested-values
  (with-hf-directory (directory)
    (write-hf-model directory "llama" (spec*) (weight-table (spec*)))
    (dolist (key '("model" "added_tokens" "post_processor"))
      (let ((object (hf-tokenizer-fixture "llama")))
        (setf (ci::json-get object key)
              (if (equal key "added_tokens") #(0) (ci::json-object "type" "TemplateProcessing" "single" #(0 1))))
        (write-hf-json directory "tokenizer.json" object)
        (local:with-weights (w directory)
          (signals local:tokenizer-error (local:load-tokenizer w)))))))
