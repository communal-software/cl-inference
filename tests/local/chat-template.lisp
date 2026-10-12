(in-package #:cl-inference/local/tests)

(in-suite :cl-inference/local)

;;; Expected prompts come from the official Jinja templates through
;;; reference/chat-templates.py, in tests/local/chat-template-expected.lisp.

(defparameter *weather-tool*
  '(:name :get_weather :summary "Get the current weather"
    :params ((:city string :required t :doc "City name")
             (:unit (member :celsius :fahrenheit))
             (:days (integer 1 7)))))

(defparameter *system* '(:role :system :content "You are a helpful assistant."))
(defparameter *user* '(:role :user :content "What is the weather in Paris?"))
(defparameter *user2* '(:role :user :content "And in Rome?"))

(defun call (&rest arguments)
  `(:id "call_1" :name :get_weather :arguments ,arguments))

(defun assistant (content &rest calls)
  `(:role :assistant :content ,content ,@(when calls `(:tool-calls ,calls))))

(defun tool-result (content)
  `(:role :tool :tool-call-id "call_1" :content ,content))

(defparameter *paris* (call :city "Paris" :days 3))
(defparameter *rome* (call :city "Rome" :threshold 0.25))
(defparameter *tools* (list *weather-tool*))

;;; (name messages . render-chat keys)
(defparameter *common-chat-cases*
  `(("plain" (,*system* ,*user*))
    ("plain-gen" (,*system* ,*user*) :add-generation-prompt t)
    ("no-system-gen" (,*user*) :add-generation-prompt t)
    ("multi-turn-gen" (,*system* ,*user* ,(assistant "It is sunny.") ,*user2*)
     :add-generation-prompt t)
    ("tools-gen" (,*system* ,*user*) :add-generation-prompt t :tools ,*tools*)
    ("tools-no-system-gen" (,*user*) :add-generation-prompt t :tools ,*tools*)
    ("tool-result-gen"
     (,*system* ,*user* ,(assistant "" *paris*) ,(tool-result "sunny, 21 C"))
     :add-generation-prompt t :tools ,*tools*)
    ("escaping"
     ((:role :system :content ,(format nil "  Be brief.  ~%"))
      (:role :user :content ,(format nil "~%  héllo 日本  "))
      ,(assistant "" (call :city (format nil "Zürich \"CH\"~%") :unit :celsius))
      ,(tool-result (format nil "say \"hi\"~%there \\ ✓")))
     :add-generation-prompt t :tools ,*tools*)))

(defparameter *llama3-chat-cases*
  `(,@*common-chat-cases*
    ("date" (,*system* ,*user*) :add-generation-prompt t :date "01 Jan 2030")))

(defparameter *qwen3-chat-cases*
  `(,@*common-chat-cases*
    ("two-results"
     (,*system* ,*user* ,(assistant "Checking." *paris* *rome*) ,(tool-result "sunny")
      ,(tool-result "rainy"))
     :add-generation-prompt t :tools ,*tools*)
    ("call-without-content"
     (,*user* ,(assistant "" *rome*) ,(tool-result "rainy"))
     :add-generation-prompt t :tools ,*tools*)
    ("thinking-history"
     (,*user* ,(assistant (format nil "<think>~%Paris is in France.~%</think>~%~%It is sunny."))
      ,*user2* ,(assistant (format nil "<think>~%Rome is in Italy.~%</think>~%~%It is rainy."))))
    ("thinking-history-gen"
     (,*user* ,(assistant (format nil "<think>~%Paris is in France.~%</think>~%~%It is sunny."))
      ,*user2*)
     :add-generation-prompt t)
    ("thinking-after-tool"
     (,*user* ,(assistant (format nil "<think>~%Need weather.~%</think>~%~%") *paris*)
      ,(tool-result "sunny")
      ,(assistant (format nil "<think>~%Got it.~%</think>~%~%It is sunny.")))
     :tools ,*tools*)
    ("last-assistant-without-thinking" (,*user* ,(assistant "Hi.")))
    ("assistant-without-thinking-after-tool"
     (,*user* ,(assistant "" *paris*) ,(tool-result "sunny") ,(assistant "Done."))
     :tools ,*tools*)
    ("thinking-off-gen" (,*system* ,*user*) :add-generation-prompt t :enable-thinking nil)))

(defun check-chat-cases (prefix class cases)
  (let ((template (make-instance class)))
    (dolist (case cases)
      (destructuring-bind (name messages &rest keys) case
        (let ((expected (cdr (assoc (format nil "~A/~A" prefix name) *chat-expected*
                                    :test #'string=))))
          (is (stringp expected) "~A/~A has no expected prompt" prefix name)
          (is (equal expected (apply #'local:render-chat template messages
                                      (append keys '(:date "26 Jul 2024"))))
              "~A/~A renders differently from the Jinja template" prefix name))))))

(test llama3-chat-matches-the-jinja-template
  (check-chat-cases "llama3" 'local:llama3-chat *llama3-chat-cases*))

(test qwen3-chat-matches-the-jinja-template
  (check-chat-cases "qwen3" 'local:qwen3-chat *qwen3-chat-cases*))

(test llama3-chat-dates-the-prompt-today-by-default
  (is (search "Today Date: " (local:render-chat (make-instance 'local:llama3-chat)
                                                (list *user*))))
  (is (not (search "26 Jul 2024" (local:render-chat (make-instance 'local:llama3-chat)
                                                    (list *user*))))))

(test llama3-chat-rejects-what-its-template-rejects
  (let ((template (make-instance 'local:llama3-chat)))
    (signals local:chat-template-error
      (local:render-chat template (list *user* (assistant "" *paris* *rome*))))
    (signals local:chat-template-error
      (local:render-chat template '() :tools *tools*))
    (signals local:chat-template-error
      (local:render-chat template (list *system*) :tools *tools*))))

(test chatml-wraps-each-turn
  (let ((template (make-instance 'local:chatml)))
    (is (equal (format nil "<|im_start|>system~%S<|im_end|>~%<|im_start|>user~%U<|im_end|>~%~
                            <|im_start|>assistant~%")
               (local:render-chat template '((:role :system :content "S")
                                             (:role :user :content "U"))
                                  :add-generation-prompt t)))
    (is (equal (format nil "<|im_start|>user~%ab<|im_end|>~%")
               (local:render-chat template
                                  '((:role :user :content ((:type :text :text "a")
                                                           (:type :text :text "b")))))))))

(test chatml-rejects-tools
  (let ((template (make-instance 'local:chatml)))
    (signals local:chat-template-error (local:render-chat template (list *user*) :tools *tools*))
    (signals local:chat-template-error (local:render-chat template (list (tool-result "x"))))
    (signals local:chat-template-error
      (local:render-chat template (list (assistant "" *paris*))))))

;;; JSON text

(test python-json-writes-floats-as-python-does
  (loop for (value text) in '((1.5 "1.5") (1f-5 "1e-05") (1f16 "1e+16") (100.0 "100.0")
                              (0.1 "0.1") (1d15 "1000000000000000.0") (1.2345f20 "1.2345e+20")
                              (0.0001 "0.0001") (2.5f-7 "2.5e-07") (-3.0 "-3.0") (0.0 "0.0")
                              (123456789012345d0 "123456789012345.0") (0.25 "0.25")
                              (1234.5 "1234.5") (1d100 "1e+100"))
        do (is (equal text (local::python-json value)) "~S" value)))

(test python-json-escapes-strings-as-python-does
  (is (equal "\"a\\\"b\\\\c\\nd\\te\\u0001 é 日\""
             (local::python-json (format nil "a\"b\\c~%d~Ce~C é 日" #\Tab (code-char 1)))))
  (is (equal "\"\\b\\f\\r\"" (local::python-json (coerce '(#\Backspace #\Page #\Return)
                                                         'string)))))

(test python-json-lays-out-containers-as-python-does
  (let ((object (ci:json-object "a" 1 "b" (vector 1 2 "x") "c" nil "d" t "e" 'null
                                "f" (ci:json-object) "g" (vector))))
    (is (equal "{\"a\": 1, \"b\": [1, 2, \"x\"], \"c\": false, \"d\": true, \"e\": null, \"f\": {}, \"g\": []}"
               (local::python-json object)))
    (is (equal (format nil "{~%    \"a\": 1,~%    \"b\": [~%        1,~%        2,~%        \"x\"~%    ],~
                            ~%    \"c\": false,~%    \"d\": true,~%    \"e\": null,~%    \"f\": {},~
                            ~%    \"g\": []~%}")
               (local::python-json object :indent 4)))))

(test python-json-rejects-what-json-cannot-hold
  (signals local:chat-template-error (local::python-json 'symbol)))

;;; Definitions used by the tests below

(local:define-chat-template test-parent-chat ()
  (:detect "AAA")
  (:message "~a:~a|")
  (:generation-prompt "go")
  (:stop "x" "y"))

(local:define-chat-template test-child-chat (test-parent-chat)
  (:detect "AAA" "BBB")
  (:stop "z"))

(test chat-template-sections-inherit-by-name
  (let ((child (make-instance 'test-child-chat)))
    (is (equal '("AAA" "BBB") (local::chat-template-entry child :detect)))
    (is (equal '("~a:~a|") (local::chat-template-entry child :message)))
    (is (equal '("z") (local::chat-template-entry child :stop)))
    (is (equal "user:hi|go" (local:render-chat child '((:role :user :content "hi"))
                                               :add-generation-prompt t)))))

(test redefining-a-chat-template-updates-its-children
  (let ((child (make-instance 'test-child-chat)))
    (unwind-protect
         (progn
           (local:define-chat-template test-parent-chat ()
             (:detect "AAA") (:message "~a=~a|") (:generation-prompt "go") (:stop "x" "y"))
           (is (equal "user=hi|" (local:render-chat child '((:role :user :content "hi"))))))
      (local:define-chat-template test-parent-chat ()
        (:detect "AAA") (:message "~a:~a|") (:generation-prompt "go") (:stop "x" "y")))))

(test excluded-markers-veto-a-match
  (is (eq 'local:chatml (loaded-chat-template-class :template "<|im_start|>")))
  (signals local:unknown-chat-template
    (loaded-chat-template-class :template "<|im_start|> <function=x>"))
  (signals local:unknown-chat-template
    (loaded-chat-template-class :template "<|im_start|> multi_step_tool <function=x>")))

(test chat-template-definitions-are-checked
  (dolist (sections '(((:unknown "x"))
                      ((:message))
                      ((:message "~a" "~a"))
                      ((:message "~a~a~a"))
                      ((:message "~Q"))
                      ((:generation-prompt "~Q"))
                      ((:detect 1))
                      ((:stop))
                      ((:stop "a") (:stop "b"))
                      (:message)))
    (signals local:chat-template-error
      (macroexpand-1 `(local:define-chat-template test-bad-chat () ,@sections)))))

;;; Selection and loading

(defun chat-file (&key template (architecture "unregistered"))
  (build-gguf :pad nil
              :metadata `(("general.architecture" 8 ,architecture)
                          ,@(when template `(("tokenizer.chat_template" 8 ,template))))))

(defun loaded-chat-template-class (&rest options)
  (with-gguf-file (path (apply #'chat-file options))
    (local:with-weights (weights path)
      (type-of (local:load-chat-template weights)))))

(local:define-architecture chat-fallback-architecture ()
  (:format :gguf (:arch "chatfallback") (:chat-template chatml)))

(local:define-architecture chat-private-architecture ()
  (:format :gguf (:arch "chatprivate") (:chat-template test-child-chat)))

(local:define-architecture chat-misspelt-architecture ()
  (:format :gguf (:arch "chatmisspelt") (:chat-template no-such-chat-template)))

(test chat-template-is-detected-from-the-files-template
  (is (eq 'local:llama3-chat
          (loaded-chat-template-class
           :template "{{ '<|start_header_id|>' }} Cutting Knowledge Date: December 2023")))
  (is (eq 'local:chatml (loaded-chat-template-class :template "{{ '<|im_start|>' }}")))
  (is (eq 'local:qwen3-chat
          (loaded-chat-template-class :template "<|im_start|> ... multi_step_tool"))))

(test chat-template-falls-back-to-the-architectures
  (is (eq 'local:chatml (loaded-chat-template-class :architecture "chatfallback")))
  (is (eq 'local:chatml (loaded-chat-template-class :architecture "chatfallback"
                                                    :template "{{ unrecognised }}")))
  (is (eq 'test-child-chat (loaded-chat-template-class :architecture "chatprivate"))))

(test chat-template-detection-beats-the-architectures
  (is (eq 'local:qwen3-chat
          (loaded-chat-template-class :architecture "chatfallback"
                                      :template "<|im_start|> multi_step_tool"))))

(test chat-template-loading-signals-when-nothing-applies
  (signals local:unknown-chat-template (loaded-chat-template-class))
  (signals local:unknown-chat-template (loaded-chat-template-class :template "{{ x }}"))
  (signals local:unknown-chat-template
    (loaded-chat-template-class :architecture "nosucharchitecture"))
  (signals local:chat-template-error (loaded-chat-template-class :architecture "chatmisspelt")))

(test chat-template-loading-reports-the-source
  (handler-case (loaded-chat-template-class :template "{{ x }}")
    (local:unknown-chat-template (condition)
      (is (equal "{{ x }}" (local:unknown-chat-template-source condition))))))

;;; llama.cpp's models/templates, kept in the gitignored reference/templates directory.

(test official-templates-are-detected
  (let ((directory (fixture-path "templates/")))
    (if (not (probe-file directory))
        (skip "reference/templates is absent")
        (loop for (file class) in '(("meta-llama-Llama-3.2-3B-Instruct" local:llama3-chat)
                                    ("meta-llama-Llama-3.1-8B-Instruct" local:llama3-chat)
                                    ("Qwen-Qwen3-0.6B" local:qwen3-chat)
                                    ("Qwen-QwQ-32B" local:chatml)
                                    ("Qwen3.5-4B" nil)
                                    ("Qwen3-Coder" nil))
              for path = (fixture-path (format nil "templates/~A.jinja" file))
              when (probe-file path)
                do (is (eq class (let ((found (local::detected-chat-template-class (fixture-text path))))
                          (and found (class-name found))))
                       "~A" file)))))

(test real-vocab-fixture-declares-chatml
  (let ((path (fixture-path "ggml-vocab-qwen2.gguf")))
    (if (not (probe-file path))
        (skip "ggml-vocab-qwen2.gguf is not in reference/")
        (local:with-weights (weights path)
          (is (typep (local:load-chat-template weights) 'local:chatml))))))

;;; Prompts as token ids

(defparameter *chat-tokens*
  '("<|im_start|>" "<|im_end|>" "<|start_header_id|>" "<|end_header_id|>" "<|eot_id|>"
    "<|end_of_text|>" "<|endoftext|>"))

(defmacro with-chat-tokenizer ((var &rest file-options) &body body)
  `(with-tokenizer (,var :tokens (append (fixture-tokens) *chat-tokens*)
                         :types (append (fixture-types) (make-list 7 :initial-element 3))
                         ,@file-options)
     ,@body))

;;; Ids: 264 <s>, 265 </s>, 267 <|im_start|>, 268 <|im_end|>, 269 <|start_header_id|>,
;;; 270 <|end_header_id|>, 271 <|eot_id|>, 272 <|end_of_text|>, 273 <|endoftext|>.

(test encode-chat-turns-control-text-into-single-ids
  (with-chat-tokenizer (tokenizer :specials '(("bos_token_id" 264) ("eos_token_id" 265)
                                              ("eot_token_id" 271)))
    (let* ((template (make-instance 'local:llama3-chat))
           (messages (list *user*))
           (ids (local:encode-chat template tokenizer messages :date "01 Jan 2030")))
      (is (= 264 (aref ids 0)))
      (is (= 1 (count 264 ids)))
      (is (= 2 (count 271 ids)))
      (is (= 3 (count 269 ids)))
      (is (equal (concatenate 'string "<s>"
                              (local:render-chat template messages :date "01 Jan 2030"
                                                                   :add-generation-prompt t))
                 (local:decode tokenizer ids :special t))))))

(test encode-chat-follows-the-files-bos-setting
  (with-chat-tokenizer (tokenizer :add-bos nil :pre "qwen2"
                                  :specials '(("eos_token_id" 265)))
    (let ((ids (local:encode-chat (make-instance 'local:chatml) tokenizer (list *user*))))
      (is (= 267 (aref ids 0)))
      (is (zerop (count 264 ids))))))

(test encode-chat-adds-the-generation-prompt-unless-told-not-to
  (with-chat-tokenizer (tokenizer :add-bos nil :pre "qwen2")
    (let ((template (make-instance 'local:chatml)))
      (is (< (length (local:encode-chat template tokenizer (list *user*)
                                        :add-generation-prompt nil))
             (length (local:encode-chat template tokenizer (list *user*))))))))

(test stop-ids-join-the-templates-tokens-and-the-files-end-tokens
  (with-chat-tokenizer (tokenizer :specials '(("bos_token_id" 264) ("eos_token_id" 265)
                                              ("eot_token_id" 271)))
    (is (equal '(271 272 265) (local:chat-template-stop-ids (make-instance 'local:llama3-chat)
                                                            tokenizer)))
    (is (equal '(268 273 265 271) (local:chat-template-stop-ids (make-instance 'local:chatml)
                                                                tokenizer)))))
