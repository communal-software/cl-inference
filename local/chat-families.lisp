(in-package #:cl-inference/local)

(define-chat-template qwen25-chat (chatml)
  (:detect "<|im_start|>" "You are Qwen, created by Alibaba Cloud." "tool_call.arguments | tojson")
  (:exclude "<function=" "multi_step_tool" "enable_thinking"))

(define-chat-template qwq-chat (chatml)
  (:detect "<|im_start|>" "message.content.split('</think>')" "enable_thinking")
  (:exclude "<function=" "multi_step_tool"))

(define-chat-template qwen3-coder-chat ()
  (:detect "<|im_start|>" "<function=" "render_extra_keys")
  (:exclude "render_content")
  (:message "<|im_start|>~a~%~a<|im_end|>~%")
  (:generation-prompt "<|im_start|>assistant~%")
  (:stop "<|im_end|>" "<|endoftext|>"))

(define-chat-template qwen35-chat (qwen3-coder-chat)
  (:detect "<|im_start|>" "<function=" "render_content" "multi_step_tool")
  (:exclude "render_extra_keys"))

(define-chat-template hermes-chat (chatml)
  (:detect "<|im_start|>" "pydantic model json schema" "json_to_python_type")
  (:exclude "<function="))

(define-chat-template llama30-chat (llama3-chat)
  (:detect "<|start_header_id|>" "add_generation_prompt" "bos_token")
  (:exclude "Cutting Knowledge Date" "builtin_tools"))

(define-chat-template llama31-chat (llama3-chat)
  (:detect "<|start_header_id|>" "Cutting Knowledge Date" "builtin_tools"))

(define-chat-template r1-chat ()
  (:detect "<｜User｜>" "<｜Assistant｜>" "tool['function']['arguments'] | tojson")
  (:generation-prompt "<｜Assistant｜><think>~%")
  (:stop "<｜end▁of▁sentence｜>"))

(define-chat-template r1-llama-chat (r1-chat)
  (:detect "<｜User｜>" "<｜Assistant｜>" "message['content'] is none")
  (:exclude "tool['function']['arguments'] | tojson"))

(defun json-call-segments (call tools)
  (append (markup (format nil "<tool_call>~%"))
          (chat-text (format nil "{\"name\": ~A, \"arguments\": ~A}"
                             (python-json (ci:wire-tool-name (getf call :name)))
                             (call-json call tools)))
          (markup (format nil "~%</tool_call>"))))

(defun grouped-tool-results (template tail &key trim trailing-newline)
  (render-turn template "user"
               (loop for response in tail
                     while (eq (getf response :role) :tool)
                     for first = t then nil
                     append (append
                             (unless (or first trailing-newline) (markup (string #\Newline)))
                             (markup (format nil "<tool_response>~%"))
                             (chat-text (if trim (trim-whitespace (message-text response))
                                            (message-text response)))
                             (markup (format nil "~%</tool_response>"))
                             (when trailing-newline (markup (string #\Newline)))))))

(defmethod render-chat ((template qwen25-chat) messages
                        &key tools add-generation-prompt &allow-other-keys)
  (render-qwen25-chat template messages tools add-generation-prompt))

(defmethod render-chat ((template qwq-chat) messages
                        &key tools add-generation-prompt (enable-thinking t) &allow-other-keys)
  (render-qwen25-chat template messages tools add-generation-prompt :qwq t
                      :enable-thinking enable-thinking))

(defun render-qwen25-chat (template messages tools generation &key qwq (enable-thinking t))
  (let ((system (when (first-message-system-p messages) (first messages))))
    (chat-rendering
     (append
      (when (or system tools (not qwq))
        (render-turn template "system"
                     (append
                      (chat-text (if system (message-text system)
                                     (if qwq "" "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.")))
                      (when tools
                        (append (markup (format nil "~%~%")) (markup *qwen3-tools-header*)
                                (loop for definition in (tool-definitions tools)
                                      append (append (markup (string #\Newline)) (chat-text definition)))
                                (markup *qwen3-tools-footer*))))))
      (loop for tail on messages for message = (first tail) for index from 0
            for role = (getf message :role)
            append
            (cond
              ((and (eq role :system) (zerop index)) nil)
              ((eq role :tool)
               (unless (and (plusp index) (eq (getf (nth (1- index) messages) :role) :tool))
                 (grouped-tool-results template tail)))
              ((eq role :assistant)
               (let* ((original (message-text message))
                      (content (if (and qwq (rest tail))
                                   (string-left-trim '(#\Newline)
                                                     (subseq original (let ((end (search "</think>" original :from-end t)))
                                                                        (if end (+ end 8) 0))))
                                   original))
                      (calls (getf message :tool-calls)))
                 (if calls
                     (append (markup "<|im_start|>assistant")
                             (when (plusp (length original))
                               (append (markup (string #\Newline)) (chat-text content)))
                             (loop for call in calls append
                                   (append (markup (string #\Newline)) (json-call-segments call tools)))
                             (markup (format nil "<|im_end|>~%")))
                     (render-turn template "assistant" content))))
              (t (render-turn template (message-role-name message) (message-text message)))))
      (when generation
        (append (render-generation-prompt template)
                (when qwq
                  (append (markup (format nil "<think>~%"))
                          (unless enable-thinking (markup "</think>"))))))))))

(defmethod render-chat ((template llama30-chat) messages
                        &key tools add-generation-prompt &allow-other-keys)
  (when (or tools (some (lambda (message) (or (getf message :tool-calls)
                                             (eq (getf message :role) :tool))) messages))
    (chat-template-fail "Llama 3.0 does not support tools"))
  (chat-rendering
   (append (loop for message in messages append
                 (render-turn template (message-role-name message) (trim-whitespace (message-text message))))
           (when add-generation-prompt (render-generation-prompt template)))))

(defun checked-builtin-tools (builtins tools)
  (unless (and (alexandria:proper-list-p builtins)
               (every (lambda (name) (and (keywordp name) (find name tools :key (lambda (tool) (getf tool :name)))))
                      builtins)
               (= (length builtins) (length (remove-duplicates builtins))))
    (chat-template-fail "Builtin tools must be distinct names of declared tools"))
  builtins)

(defun render-llama31-call (message tools builtins builtin-mode)
  (let ((calls (getf message :tool-calls)))
    (unless (= 1 (length calls)) (chat-template-fail "Llama 3.1 needs one tool call per message"))
    (let* ((call (first calls)) (name (getf call :name))
           (arguments (ci:json-object-alist (ci:call-arguments->json call tools))))
      (append
       (markup (format nil "<|start_header_id|>assistant<|end_header_id|>~%~%"))
       (if (member name builtins)
           (append
            (markup "<|python_tag|>")
            (chat-text (format nil "~A.call(~{~A~^, ~})" (ci:wire-tool-name name)
                               (loop for (key . value) in arguments
                                     do (unless (and (stringp value) (notany (lambda (c) (find c '(#\" #\\ #\Newline #\Return))) value))
                                          (chat-template-fail "Llama builtin arguments need strings without quotes, backslashes or line breaks"))
                                     collect (format nil "~A=\"~A\"" key value)))))
           (chat-text (format nil "{\"name\": ~A, \"parameters\": ~A}"
                              (python-json (ci:wire-tool-name name)) (call-json call tools))))
       (markup (if builtin-mode "<|eom_id|>" "<|eot_id|>"))))))

(defmethod render-chat ((template llama31-chat) messages
                        &key tools add-generation-prompt (date (current-date-string))
                          (builtin-tools nil builtin-mode) (tools-in-user-message t)
                        &allow-other-keys)
  (checked-builtin-tools builtin-tools tools)
  (let* ((system (when (first-message-system-p messages) (pop messages)))
         (definitions (loop for definition in (tool-definitions tools 4)
                            append (append (chat-text definition) (markup (format nil "~%~%")))))
         (segments
           (render-turn template "system"
                        (append
                         (when (or tools builtin-mode) (markup (format nil "Environment: ipython~%")))
                         (when builtin-mode
                           (append (markup "Tools: ")
                                   (chat-text (format nil "~{~A~^, ~}" (mapcar #'ci:wire-tool-name
                                                 (remove :code_interpreter builtin-tools))))
                                   (markup (format nil "~%~%"))))
                         (markup (format nil "Cutting Knowledge Date: December 2023~%Today Date: "))
                         (chat-text date) (markup (format nil "~%~%"))
                         (when (and tools (not tools-in-user-message))
                           (append (markup (format nil "You have access to the following functions. To call a function, please respond with JSON for a function call.Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.Do not use variables.~%~%"))
                                   definitions))
                         (chat-text (if system (trim-whitespace (message-text system)) ""))))))
    (when (and tools tools-in-user-message)
      (unless messages (chat-template-fail "Cannot put tools in a missing first user message"))
      (setf segments
            (append segments
                    (render-turn template "user"
                                 (append
                                  (markup (format nil "Given the following functions, please respond with a JSON for a function call with its proper arguments that best answers the given prompt.~%~%Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.Do not use variables.~%~%"))
                                  definitions (chat-text (trim-whitespace (message-text (pop messages)))))))))
    (chat-rendering
     (append segments
             (loop for message in messages append
                   (if (getf message :tool-calls)
                       (render-llama31-call message tools builtin-tools builtin-mode)
                       (render-llama3-message template message tools)))
             (when add-generation-prompt (render-generation-prompt template))))))

(defparameter *coder-tools-footer* "

If you choose to call a tool ONLY reply in the following format with NO suffix:

<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
<parameter=example_parameter_2>
value_2
</parameter>
</function>
</tool_call>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format: the tool calling block MUST begin with an opening <tool_call> tag and end with a closing </tool_call> tag.
- Required parameters MUST be specified
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
</IMPORTANT>")

(defparameter *qwen35-tools-footer* "

If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>
</tool_call>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags
- Required parameters MUST be specified
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
</IMPORTANT>")

(defparameter *hermes-tools-header* "You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions. Here are the available tools: <tools> ")

(defparameter *hermes-tools-footer* " </tools>Use the following pydantic model json schema for each tool call you will make: {\"properties\": {\"name\": {\"title\": \"Name\", \"type\": \"string\"}, \"arguments\": {\"title\": \"Arguments\", \"type\": \"object\"}}, \"required\": [\"name\", \"arguments\"], \"title\": \"FunctionCall\", \"type\": \"object\"}}
For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:
<tool_call>
{\"name\": <function-name>, \"arguments\": <args-dict>}
</tool_call><|im_end|>
")

(defun object-fields (object)
  (ci:json-object-alist object))

(defun python-value (value)
  (typecase value
    (string value)
    (float (python-float value))
    (integer (princ-to-string value))
    (t (case value ((t) "True") ((nil) "False") (null "None")
         (t (python-json value))))))

(defun xml-field (name value)
  (append (markup (format nil "~%<~A>" name)) (chat-text value)
          (markup (format nil "</~A>" name))))

(defun xml-extra-fields (object handled)
  (loop for (key . value) in (object-fields object)
        unless (member key handled :test #'string=)
          append (xml-field key (python-value value))))

(defun coder-tool-definition (tool)
  (let* ((function (cdr (assoc "function" (object-fields tool) :test #'string=)))
         (fields (object-fields function))
         (parameters (cdr (assoc "parameters" fields :test #'string=)))
         (properties (cdr (assoc "properties" (object-fields parameters) :test #'string=))))
    (append
     (markup (format nil "~%<function>"))
     (xml-field "name" (cdr (assoc "name" fields :test #'string=)))
     (when (assoc "description" fields :test #'string=)
       (xml-field "description" (trim-whitespace (cdr (assoc "description" fields :test #'string=)))))
     (markup (format nil "~%<parameters>"))
     (loop for (name . schema) in (object-fields properties)
           for fields = (object-fields schema)
           append (append
                   (markup (format nil "~%<parameter>")) (xml-field "name" name)
                   (when (assoc "type" fields :test #'string=)
                     (xml-field "type" (let ((type (cdr (assoc "type" fields :test #'string=))))
                                        (if (stringp type) type
                                            (format nil "[~{~A~^, ~}]"
                                                    (map 'list (lambda (item) (format nil "'~A'" item)) type))))))
                   (when (assoc "description" fields :test #'string=)
                     (xml-field "description" (trim-whitespace (cdr (assoc "description" fields :test #'string=)))))
                   (xml-extra-fields schema '("name" "type" "description"))
                   (markup (format nil "~%</parameter>"))))
     (xml-extra-fields parameters '("type" "properties"))
     (markup (format nil "~%</parameters>"))
     (xml-extra-fields function '("type" "name" "description" "parameters"))
     (markup (format nil "~%</function>")))))

(defun function-call-segments (call tools)
  (append (markup (format nil "<tool_call>~%<function="))
          (chat-text (ci:wire-tool-name (getf call :name))) (markup (format nil ">~%"))
          (loop for (key . value) in (object-fields (ci:call-arguments->json call tools))
                append (append (markup "<parameter=") (chat-text key) (markup (format nil ">~%"))
                               (chat-text (python-value value)) (markup (format nil "~%</parameter>~%"))))
          (markup (format nil "</function>~%</tool_call>"))))

(defmethod render-chat ((template qwen3-coder-chat) messages
                        &key tools add-generation-prompt (enable-thinking t) &allow-other-keys)
  (let* ((q35 (typep template 'qwen35-chat))
         (system (when (first-message-system-p messages) (first messages)))
         (last-query (when q35
                       (position-if (lambda (message)
                                      (and (eq (getf message :role) :user)
                                           (not (tool-response-text-p
                                                 (trim-whitespace (message-text message))))))
                                    messages :from-end t))))
    (when q35
      (unless last-query
        (chat-template-fail "Qwen3.5 needs a user query"))
      (when (some (lambda (message) (eq (getf message :role) :system)) (rest messages))
        (chat-template-fail "System message must be first")))
    (chat-rendering
     (append
      (when (or system tools)
        (render-turn
         template "system"
         (if q35
             (append
              (when tools
                (append (markup (format nil "# Tools~%~%You have access to the following functions:~%~%<tools>"))
                        (loop for definition in (tool-definitions tools) append
                              (append (markup (string #\Newline)) (chat-text definition)))
                        (markup (format nil "~%</tools>")) (markup *qwen35-tools-footer*)))
              (when system
                (append (when (and tools (plusp (length (trim-whitespace (message-text system)))))
                          (markup (format nil "~%~%")))
                        (chat-text (trim-whitespace (message-text system))))))
             (append
              (chat-text (if system (message-text system)
                             "You are Qwen, a helpful AI assistant that can interact with a computer to solve tasks."))
              (when tools
                (append (markup (format nil "~%~%# Tools~%~%You have access to the following tools:~%~%<tools>"))
                        (loop for tool across (ci:tools->json tools) append (coder-tool-definition tool))
                        (markup (format nil "~%</tools>")) (markup *coder-tools-footer*)))))))
      (loop for tail on messages for message = (first tail) for index from 0
            for role = (getf message :role)
            for content = (if q35 (trim-whitespace (message-text message)) (message-text message))
            append
            (cond
              ((and system (zerop index)) nil)
              ((eq role :tool)
               (unless (and (plusp index) (eq (getf (nth (1- index) messages) :role) :tool))
                 (grouped-tool-results template tail :trim q35 :trailing-newline (not q35))))
              ((eq role :assistant)
               (let ((calls (getf message :tool-calls)))
                 (multiple-value-bind (text reasoning) (if q35 (split-reasoning content) (values content ""))
                   (if (or q35 calls)
                       (append
                        (markup "<|im_start|>assistant")
                        (if q35
                            (append (markup (string #\Newline))
                                    (when (> index last-query)
                                      (append (markup (format nil "<think>~%"))
                                              (chat-text (trim-whitespace reasoning))
                                              (markup (format nil "~%</think>~%~%"))))
                                    (chat-text text))
                            (when (plusp (length (trim-whitespace text)))
                              (append (markup (string #\Newline)) (chat-text (trim-whitespace text))
                                      (markup (string #\Newline)))))
                        (loop for call in calls for first = t then nil append
                              (append
                               (cond ((not q35) (markup (string #\Newline)))
                                     ((not first) (markup (string #\Newline)))
                                     ((plusp (length (trim-whitespace text))) (markup (format nil "~%~%"))))
                               (function-call-segments call tools)))
                        (markup (format nil "<|im_end|>~%")))
                       (render-turn template "assistant" content)))))
              (t (render-turn template (message-role-name message) content))))
      (when add-generation-prompt
        (append (render-generation-prompt template)
                (when q35 (markup (if enable-thinking (format nil "<think>~%")
                                      (format nil "<think>~%~%</think>~%~%"))))))))))

(defun hermes-python-type (schema)
  (let* ((fields (when (typep schema 'ci:json-object) (object-fields schema)))
         (type (cdr (assoc "type" fields :test #'string=))))
    (cond
      ((null type) "Union[]")
      ((and (vectorp type) (not (stringp type)))
       (format nil "Union[~{~A~^,~}]"
               (loop for item across type collect (hermes-python-type (ci:json-object "type" item)))))
      ;; The pinned Jinja macro applies items to the schema, yielding Union[].
      ((string= type "array") "list[Union[]]")
      ((string= type "object")
       (let ((additional (assoc "additionalProperties" fields :test #'string=)))
         (if additional (format nil "dict[str, ~A]" (hermes-python-type (cdr additional))) "dict")))
      (t (or (cdr (assoc type '(("string" . "str") ("number" . "float")
                               ("integer" . "int") ("boolean" . "bool")) :test #'string=))
             (chat-template-fail "The pinned Hermes template cannot describe type ~S" type))))))

(defun hermes-tool-definition (tool)
  (let* ((fields (object-fields (cdr (assoc "function" (object-fields tool) :test #'string=))))
         (name (cdr (assoc "name" fields :test #'string=)))
         (parameters (cdr (assoc "parameters" fields :test #'string=)))
         (properties (object-fields (cdr (assoc "properties" (object-fields parameters) :test #'string=)))))
    (with-output-to-string (out)
      (format out "{\"type\": \"function\", \"function\": {\"name\": \"~A\", \"description\": \"~A(~{~A~^, ~}) - ~A~%~%"
              name name (loop for (key . schema) in properties collect (format nil "~A: ~A" key (hermes-python-type schema)))
              (or (cdr (assoc "description" fields :test #'string=)) ""))
      (when properties
        (format out "    Args:~%")
        (loop for (key . schema) in properties
              do (format out "        ~A(~A): ~A" key (hermes-python-type schema)
                         (trim-whitespace (or (cdr (assoc "description" (object-fields schema) :test #'string=)) "")))))
      (format out "\", \"parameters\": ~A}" (if properties (python-json parameters) "{}")))))

(defmethod render-chat ((template hermes-chat) messages
                        &key tools add-generation-prompt &allow-other-keys)
  (chat-rendering
   (append
    (markup (format nil "<|im_start|>system~%")) (markup *hermes-tools-header*)
    (loop for tool across (ci:tools->json tools) for first = t then nil append
          (append (unless first (markup (string #\Newline))) (chat-text (hermes-tool-definition tool))))
    (markup *hermes-tools-footer*)
    (loop for tail on messages for message = (first tail) for index from 0
          for role = (getf message :role) for calls = (getf message :tool-calls)
          append
          (cond
            ((eq role :tool)
             (append
              (when (and (plusp index) (not (eq (getf (nth (1- index) messages) :role) :tool)))
                (markup (format nil "<|im_start|>tool~%")))
              (markup (format nil "<tool_response>~%")) (chat-text (message-text message))
              (markup (if (rest tail) (format nil "~%</tool_response>~%") (format nil "~%</tool_response>")))
              (when (or (null (rest tail)) (not (eq (getf (second tail) :role) :tool))) (markup "<|im_end|>"))))
            (calls
             (append (markup "<|im_start|>assistant")
                     (loop for call in calls append (append (markup (string #\Newline)) (json-call-segments call tools)))
                     (markup (format nil "<|im_end|>~%"))))
            (t (render-turn template (message-role-name message) (message-text message)))))
    (when add-generation-prompt (render-generation-prompt template)))))

(defmethod render-chat ((template r1-chat) messages
                        &key tools add-generation-prompt (enable-thinking t) &allow-other-keys)
  (let ((is-tool nil) (first-call t) (first-output t)
        (system (find :system messages :key (lambda (message) (getf message :role)) :from-end t)))
    (chat-rendering
     (append
      (when system (chat-text (message-text system)))
      (loop for message in messages for role = (getf message :role)
            for calls = (getf message :tool-calls) append
            (case role
              (:user (setf is-tool nil)
                     (append (markup "<｜User｜>") (chat-text (message-text message))))
              (:assistant
               (append
                (when calls
                  (setf is-tool nil)
                  (append
                   (loop for call in calls
                         for initial = first-call
                         do (setf first-call nil)
                         append
                         (append
                          (markup (if initial "<｜Assistant｜><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>"
                                      (format nil "~%<｜tool▁call▁begin｜>function<｜tool▁sep｜>")))
                          (chat-text (ci:wire-tool-name (getf call :name)))
                          (markup (format nil "~%```json~%")) (chat-text (call-json call tools))
                          (markup (format nil "~%```<｜tool▁call▁end｜>"))
                          (when (and (typep template 'r1-llama-chat) (not initial))
                            (markup "<｜tool▁calls▁end｜><｜end▁of▁sentence｜>"))))
                   (unless (typep template 'r1-llama-chat)
                     (markup "<｜tool▁calls▁end｜><｜end▁of▁sentence｜>"))))
                (unless (and calls (null (getf message :content)))
                  (let ((content (message-text message)))
                    (prog1
                        (append
                         (markup (if is-tool "<｜tool▁outputs▁end｜>" "<｜Assistant｜>"))
                         (chat-text (if is-tool content
                                        (subseq content (let ((end (search "</think>" content :from-end t)))
                                                          (if end (+ end 8) 0)))))
                         (markup "<｜end▁of▁sentence｜>"))
                      (setf is-tool nil))))))
              (:tool
               (setf is-tool t)
               (prog1
                   (append (markup (if first-output "<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>"
                                       (format nil "~%<｜tool▁output▁begin｜>")))
                           (chat-text (message-text message)) (markup "<｜tool▁output▁end｜>"))
                 (setf first-output nil)))))
      (when is-tool (markup "<｜tool▁outputs▁end｜>"))
      (when (and add-generation-prompt (not is-tool))
        (append (render-generation-prompt template)
                (when (and (not enable-thinking) (not (typep template 'r1-llama-chat)))
                  (markup "</think>"))))))))

(defmethod render-chat :around ((template chat-template) messages
                               &rest keys &key builtin-tools tools-in-user-message &allow-other-keys)
  (declare (ignore builtin-tools tools-in-user-message))
  (when (and (null messages)
             (or (typep template 'llama31-chat)
                 (and (typep template 'llama3-chat) (not (typep template 'llama30-chat)))
                 (typep template 'qwen25-chat) (typep template 'qwq-chat)
                 (typep template 'qwen3-chat) (typep template 'qwen3-coder-chat)))
    (chat-template-fail "This template needs at least one message"))
  (when (and (or (loop for key in keys by #'cddr thereis (eq key :builtin-tools))
                 (loop for key in keys by #'cddr thereis (eq key :tools-in-user-message)))
             (not (typep template 'llama31-chat)))
    (chat-template-fail "Builtin tools and tool placement require a Llama 3.1 template"))
  (call-next-method))
