(in-package #:cl-inference/local)

;;; The prompt formats follow the Llama 3.2 Instruct and Qwen3 Jinja templates byte for byte.

(define-chat-template llama3-chat ()
  (:detect "<|start_header_id|>" "Cutting Knowledge Date")
  (:message "<|start_header_id|>~a<|end_header_id|>~%~%~a<|eot_id|>")
  (:generation-prompt "<|start_header_id|>assistant<|end_header_id|>~%~%")
  (:stop "<|eot_id|>" "<|eom_id|>" "<|end_of_text|>"))

(define-chat-template chatml ()
  (:detect "<|im_start|>")
  (:exclude "<function=")
  (:message "<|im_start|>~a~%~a<|im_end|>~%")
  (:generation-prompt "<|im_start|>assistant~%")
  (:stop "<|im_end|>" "<|endoftext|>"))

(define-chat-template qwen3-chat (chatml)
  (:detect "<|im_start|>" "multi_step_tool"))

(defun python-space-p (char)
  (or (space-p char) (<= #x1c (char-code char) #x1f)))

(defun trim-whitespace (string)
  "STRING without leading and trailing whitespace, as Python's str.strip."
  (let ((start (position-if-not #'python-space-p string)))
    (if start
        (subseq string start (1+ (position-if-not #'python-space-p string :from-end t)))
        "")))

(defun first-message-system-p (messages)
  (eq (getf (first messages) :role) :system))

(defun tool-definitions (tools &optional indent)
  "TOOLS as a list of JSON texts, one per tool."
  (map 'list (lambda (tool) (python-json tool :indent indent)) (ci:tools->json tools)))

(defun call-json (call tools)
  (python-json (ci:call-arguments->json call tools)))

;;; Llama 3

(defparameter *months* #("Jan" "Feb" "Mar" "Apr" "May" "Jun" "Jul" "Aug" "Sep" "Oct" "Nov" "Dec"))

(defun current-date-string ()
  (multiple-value-bind (second minute hour day month year) (get-decoded-time)
    (declare (ignore second minute hour))
    (format nil "~2,'0D ~A ~D" day (aref *months* (1- month)) year)))

(defun render-llama3-message (template message tools)
  (let ((calls (getf message :tool-calls)))
    (cond (calls
           (unless (= 1 (length calls))
             (chat-template-fail "~S renders one tool call per message, got ~D"
                                 (class-name (class-of template)) (length calls)))
           (render-turn template "assistant"
                        (format nil "{\"name\": \"~A\", \"parameters\": ~A}"
                                (ci:wire-tool-name (getf (first calls) :name))
                                (call-json (first calls) tools))))
          ((eq (getf message :role) :tool)
           (render-turn template "ipython" (python-json (message-text message))))
          (t (render-turn template (message-role-name message)
                          (trim-whitespace (message-text message)))))))

(defmethod render-chat ((template llama3-chat) messages
                        &key tools add-generation-prompt (date (current-date-string))
                        &allow-other-keys)
  (let ((system (and (first-message-system-p messages) (first messages)))
        (messages (if (first-message-system-p messages) (rest messages) messages)))
    (with-output-to-string (out)
      (write-string (render-turn template "system"
                                 (format nil "~:[~;Environment: ipython~%~]~
                                              Cutting Knowledge Date: December 2023~%~
                                              Today Date: ~A~%~%~A"
                                         tools date
                                         (if system (trim-whitespace (message-text system)) "")))
                    out)
      (when tools
        (unless messages
          (chat-template-fail "Cannot put tools in the first user message when there is none"))
        (write-string (render-turn template "user"
                                   (format nil "Given the following functions, please respond with a JSON for a function call with its proper arguments that best answers the given prompt.~%~%~
                                                Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.~
                                                Do not use variables.~%~%~
                                                ~{~A~%~%~}~A"
                                           (tool-definitions tools 4)
                                           (trim-whitespace (message-text (first messages)))))
                      out)
        (pop messages))
      (dolist (message messages)
        (write-string (render-llama3-message template message tools) out))
      (when add-generation-prompt
        (write-string (render-generation-prompt template) out)))))

;;; Qwen3

(defparameter *qwen3-tools-header*
  (format nil "# Tools~%~%You may call one or more functions to assist with the user query.~%~%~
               You are provided with function signatures within <tools></tools> XML tags:~%<tools>"))

(defparameter *qwen3-tools-footer*
  (format nil "~%</tools>~%~%For each function call, return a json object with function name and ~
               arguments within <tool_call></tool_call> XML tags:~%<tool_call>~%~
               {\"name\": <function-name>, \"arguments\": <args-json-object>}~%</tool_call>"))

(defun tool-response-text-p (text)
  (and (uiop:string-prefix-p "<tool_response>" text)
       (uiop:string-suffix-p text "</tool_response>")))

(defun last-query-index (messages)
  "Index of the last user message that is not a wrapped tool response; the last message
when there is none."
  (or (position-if (lambda (message)
                     (and (eq (getf message :role) :user)
                          (not (tool-response-text-p (message-text message)))))
                   messages :from-end t)
      (1- (length messages))))

(defun split-reasoning (content)
  "CONTENT after its last </think>, and the reasoning before it, as the Qwen3 template splits."
  (let ((end (search "</think>" content :from-end t)))
    (if (null end)
        (values content "")
        (let* ((before (string-right-trim '(#\Newline)
                                          (subseq content 0 (search "</think>" content))))
               (open (search "<think>" before :from-end t)))
          (values (string-left-trim '(#\Newline) (subseq content (+ end (length "</think>"))))
                  (string-left-trim '(#\Newline)
                                    (if open
                                        (subseq before (+ open (length "<think>")))
                                        before)))))))

(defun render-qwen3-assistant (template message tools after-query-p last-p)
  "Reasoning is kept only after the last query, and then only when the message is the last or
has some."
  (multiple-value-bind (content reasoning) (split-reasoning (message-text message))
    (render-turn template "assistant"
                 (with-output-to-string (out)
                   (if (and after-query-p (or last-p (plusp (length reasoning))))
                       (format out "<think>~%~A~%</think>~%~%~A"
                               (string-trim '(#\Newline) reasoning) content)
                       (write-string content out))
                   (loop for call in (getf message :tool-calls)
                         for first = t then nil
                         do (when (or (not first) (plusp (length content)))
                              (terpri out))
                            (format out "<tool_call>~%{\"name\": \"~A\", \"arguments\": ~A}~%</tool_call>"
                                    (ci:wire-tool-name (getf call :name))
                                    (call-json call tools)))))))

(defmethod render-chat ((template qwen3-chat) messages
                        &key tools add-generation-prompt (enable-thinking t) &allow-other-keys)
  (let ((system (and (first-message-system-p messages) (first messages)))
        (last-query (last-query-index messages))
        (previous nil))
    (with-output-to-string (out)
      (cond (tools
             (write-string (render-turn template "system"
                                        (format nil "~@[~A~%~%~]~A~{~%~A~}~A"
                                                (and system (message-text system))
                                                *qwen3-tools-header* (tool-definitions tools)
                                                *qwen3-tools-footer*))
                           out))
            (system
             (write-string (render-turn template "system" (message-text system)) out)))
      (loop for tail on messages
            for index from 0
            for message = (car tail)
            for role = (getf message :role)
            do (case role
                 (:user (write-string (render-turn template "user" (message-text message)) out))
                 (:system (unless (zerop index)
                            (write-string (render-turn template "system" (message-text message))
                                          out)))
                 (:assistant (write-string (render-qwen3-assistant template message tools
                                                                   (> index last-query)
                                                                   (null (cdr tail)))
                                           out))
                 (:tool (unless (eq previous :tool)
                          (write-string
                           (render-turn template "user"
                                        (format nil "~{~A~^~%~}"
                                                (loop for response in tail
                                                      while (eq (getf response :role) :tool)
                                                      collect (format nil "<tool_response>~%~A~%</tool_response>"
                                                                      (message-text response)))))
                           out))))
               (setf previous role))
      (when add-generation-prompt
        (write-string (render-generation-prompt template) out)
        (unless enable-thinking
          (format out "<think>~%~%</think>~%~%"))))))
