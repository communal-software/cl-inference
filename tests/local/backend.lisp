(in-package #:cl-inference/local/tests)

(in-suite :cl-inference/local)

(defun backend-fixture (template)
  (let* ((tokens (append (fixture-tokens)
                         '("<|im_start|>" "<|im_end|>" "</think>" "<|python_tag|>"
                           "<|start_header_id|>" "<|end_header_id|>" "<|eot_id|>"
                           "<|eom_id|>" "<|end_of_text|>")))
         (spec (spec* :vocab (length tokens)))
         (table (weight-table spec)))
    (maphash (lambda (name entry)
               (setf (cdr entry)
                     (loop for i below (length (cdr entry))
                           collect (if (or (string= name "token_embd.weight")
                                           (search "norm.weight" name)
                                           (and (string= name "output.weight")
                                                (= (floor i (spec-get spec :dim)) 97)))
                                       1f0 0f0))))
             table)
    (spec-gguf "llama" spec table
               `(("tokenizer.ggml.model" 8 "gpt2")
                 ("tokenizer.ggml.pre" 8 "llama-bpe")
                 ("tokenizer.ggml.tokens" 9 (:array 8 ,tokens))
                 ("tokenizer.ggml.token_type" 9 (:array 5 ,(append (fixture-types) (make-list 9 :initial-element 3))))
                 ("tokenizer.ggml.merges" 9 (:array 8 ,(fixture-merges)))
                 ("tokenizer.ggml.add_bos_token" 7 nil)
                 ("tokenizer.ggml.eos_token_id" 4 268)
                 ("tokenizer.chat_template" 8 ,template)))))

(defmacro with-local-backend ((backend &rest options) &body body)
  `(with-gguf-file (path (backend-fixture "<|im_start|>"))
     (local:with-weights (weights path)
       (let ((,backend (local:make-local-backend weights ,@options)))
         ,@body))))

(defparameter *backend-messages* '((:role :user :content "hi")))
(defparameter *parser-tools*
  '((:name :weather :params ((:city string :required t)))
    (:name :python :params ((:code string :required t)))))

(defun parser-result (template chunks &optional (tools *parser-tools*))
  (let* ((events nil)
         (parser (local:make-reply-parser (make-instance template) :tools tools :ref :turn
                                         :sink (lambda (e) (push e events)))))
    (dolist (chunk chunks) (local:reply-parser-push parser chunk))
    (multiple-value-bind (text calls reasoning) (local:reply-parser-finish parser)
      (values text calls reasoning (nreverse events)))))

(defun event-text (events)
  (format nil "~{~A~}" (loop for event in events when (eq :text-delta (getf event :type))
                             collect (getf event :text))))

(test local-backend-matches-scripted-contract
  (with-local-backend (backend)
    (let* ((events nil)
           (request (list :messages *backend-messages* :temperature 0 :max-tokens 3
                          :ref :turn :stream (lambda (e) (push e events))))
           (local (apply #'ci:complete backend request))
           (local-events (reverse events))
           (fake (apply #'ci:complete (ci:make-scripted-backend :replies '("aaa")) request)))
      (is (not (ci:result-error-p local)))
      (is (equal (getf (second local) :content) (getf (second fake) :content)))
      (dolist (key '(:role :tool-calls :done))
        (is (equal (getf (second local) key) (getf (second fake) key))))
      (is (eq :length (getf (getf (second local) :meta) :finish-reason)))
      (is (= 3 (getf (getf (getf (second local) :meta) :usage) :completion-tokens)))
      (is (equal "aaa" (event-text local-events)))
      (is (= 1 (count :done local-events :key (lambda (e) (getf e :type)))))
      (is (every (lambda (e) (eq :turn (getf e :ref))) local-events)))))

(test local-backend-reuses-or-isolates-cache
  (dolist (policy '(:serialized :per-call))
    (with-local-backend (backend :session-policy policy)
      (let ((session (local::backend-session backend)))
        (dotimes (i 2)
          (is (equal "aaa" (ci:content-text
                             (getf (ci:complete* backend :messages *backend-messages*
                                                         :temperature 0 :max-tokens 3) :content)))))
        (is (eq session (local::backend-session backend)))
        (is (eq policy (getf (ci:describe-backend backend) :session-policy)))))))

(test local-backend-stops-and-validates-input
  (with-local-backend (backend)
    (let ((reply (ci:complete* backend :messages *backend-messages* :temperature 0
                                       :max-tokens 10 :stop '("aa"))))
      (is (null (getf reply :content)))
      (is (eq :stop (getf (getf reply :meta) :finish-reason))))
    (dolist (options '((:temperature -1) (:top-p 0) (:top-k 0) (:seed "x")
                       (:max-tokens -1) (:stop ("")) (:timeout -1) (:cancel t)
                       (:tools ((:name :weather))) (:tools 1) (:tools ((:name "weather")))
                       (:tools ((:name :weather :params ((:x no-such-type)))))))
      (let ((events nil))
        (let ((result (apply #'ci:complete backend :messages *backend-messages*
                             :stream (lambda (e) (push e events)) options)))
          (is (eq :bad-request (first (ci:result-error result))))
          (is (= 1 (length events)))
          (is (equal result (getf (first events) :reason))))))
    (is (eq :bad-request
            (first (ci:result-error
                    (ci:complete backend :messages '((:role :user :content
                                                       ((:type :image :url "x")))))))))
    (signals ci:completion-bad-request
      (ci:complete* backend :messages *backend-messages* :max-tokens -1))))

(test local-backend-context-and-closed-weights
  (with-local-backend (backend :n-ctx 2)
    (is (eq :bad-request
            (first (ci:result-error (ci:complete backend :messages *backend-messages*))))))
  (with-local-backend (backend)
    (local:close-weights (local::backend-weights backend))
    (is (eq :unavailable (getf (ci:describe-backend backend) :status)))
    (is (equal '(:error :unavailable) (ci:complete backend :messages *backend-messages*)))))

(test local-backend-sampling-agrees-with-session
  (with-local-backend (backend)
    (let* ((request (list :messages *backend-messages* :temperature 0.7 :top-k 4 :top-p 0.9
                         :seed 42 :max-tokens 4))
           (session (local::backend-new-session backend))
           (expected (local:generate session *backend-messages* :max-tokens 4
                                     :sampler (local:make-sampler :temperature 0.7 :top-k 4
                                                                  :top-p 0.9 :seed 42))))
      (is (equal expected (ci:content-text (getf (second (apply #'ci:complete backend request)) :content))))
      (is (equal (apply #'ci:complete backend request) (apply #'ci:complete backend request))))))

(test parsers-preserve-results-at-every-boundary
  (dolist (case `((local:qwen3-chat ,(format nil "<think>~%private~%</think>~%~%Hello é.<tool_call>{\"name\":\"weather\",\"arguments\":{\"city\":\"Paris\"}}</tool_call>!")
                                   "Hello é.!" "private" :weather)
                 (local:llama3-chat "{\"name\":\"weather\",\"parameters\":{\"city\":\"Paris\"}}" "" nil :weather)
                 (local:llama3-chat "<|python_tag|>print('é')" "" nil :python)))
    (destructuring-bind (template output expected reasoning name) case
      (loop for split from 0 to (length output)
            do (multiple-value-bind (text calls thought events)
                   (parser-result template (list (subseq output 0 split) (subseq output split)))
                 (is (equal expected text))
                 (is (equal reasoning thought))
                 (is (= 1 (length calls)))
                 (is (eq name (getf (first calls) :name)))
                 (is (equal "local-1" (getf (first calls) :id)))
                 (is (equal expected (event-text events)))
                 (is (= 1 (count :tool-call-delta events :key (lambda (e) (getf e :type)))))
                 (is (every (lambda (e) (eq :turn (getf e :ref))) events)))))))

(test parsers-handle-multiple-tools-and-escaped-markers
  (let ((output "a<tool_call>{\"name\":\"weather\",\"arguments\":{\"city\":\"a \\\" </tool_call> é\"}}</tool_call>b<tool_call>{\"name\":\"weather\",\"arguments\":{\"city\":\"Rome\"}}</tool_call>c"))
    (multiple-value-bind (text calls reasoning events)
        (parser-result 'local:qwen3-chat (map 'list #'string output))
      (is (equal "abc" text))
      (is (null reasoning))
      (is (equal '("local-1" "local-2") (mapcar (lambda (c) (getf c :id)) calls)))
      (is (equal "a \" </tool_call> é" (getf (getf (first calls) :arguments) :city)))
      (is (equal '(:text-delta :tool-call-delta :text-delta :tool-call-delta :text-delta)
                 (mapcar (lambda (e) (getf e :type)) events))))))

(test parsers-reject-invalid-tools
  (dolist (output '("<tool_call>{}</tool_call>"
                    "<tool_call>{\"name\":\"missing\",\"arguments\":{}}</tool_call>"
                    "<tool_call>{\"name\":\"weather\",\"arguments\":[]}</tool_call>"
                    "<tool_call>{\"name\":\"weather\",\"arguments\":{}}</tool_call>"
                    "<tool_call>{\"name\":\"weather\",\"arguments\":{\"extra\":1}}</tool_call>"
                    "<tool_call>{\"name\":\"weather\"}"
                    "<tool_call>not-json</tool_call>"))
    (signals local:reply-parser-error (parser-result 'local:qwen3-chat (list output))))
  (signals local:reply-parser-error
    (parser-result 'local:llama3-chat '("<|python_tag|>print(1)") nil))
  (signals local:reply-parser-error
    (parser-result 'local:llama3-chat '("{\"name\":\"weather\"") *parser-tools*)))

(test parsers-keep-ordinary-json-and-unfinished-reasoning
  (dolist (template '(local:chatml local:llama3-chat))
    (multiple-value-bind (text calls) (parser-result template '("{\"ordinary\":1}") nil)
      (is (equal "{\"ordinary\":1}" text))
      (is (null calls))))
  (multiple-value-bind (text calls reasoning)
      (parser-result 'local:qwen3-chat '("<think>still thinking") nil)
    (is (equal "" text))
    (is (null calls))
    (is (equal "still thinking" reasoning)))
  (multiple-value-bind (text calls reasoning)
      (parser-result 'local:qwen3-chat '("<think>secret</think>hello") nil)
    (is (equal "hello" text))
    (is (null calls))
    (is (equal "secret" reasoning))))

(test local-backend-deadline-cancel-and-blocked-sinks
  (with-local-backend (backend)
    (let ((token (ci:make-cancel-token)) (events nil))
      (ci:cancel token)
      (is (equal '(:error :cancelled)
                 (ci:complete backend :messages *backend-messages* :cancel token
                                      :stream (lambda (e) (push e events)))))
      (is (= 1 (length events))))
    (is (equal '(:error :timeout)
               (ci:complete backend :messages *backend-messages* :timeout 0)))
    (let ((events nil))
      (is (equal '(:error :timeout)
                 (ci:complete backend :messages *backend-messages* :temperature 0 :max-tokens 3
                                      :timeout 100 :stream (lambda (e)
                                                             (push e events)
                                                             (when (eq :text-delta (getf e :type))
                                                               (sleep 10))))))
      (is (= 1 (count :done events :key (lambda (e) (getf e :type)))))
      (is (null (local::backend-session backend))))
    (let ((ci:*sink-grace* 0.05) (start (get-internal-real-time)))
      (is (not (ci:result-error-p
                (ci:complete backend :messages *backend-messages* :temperature 0 :max-tokens 1
                                     :stream (lambda (e) (when (eq :done (getf e :type)) (sleep 10)))))))
      (is (< (/ (- (get-internal-real-time) start) internal-time-units-per-second) 2)))
    (is (equal "a" (ci:content-text (getf (ci:complete* backend :messages *backend-messages*
                                                              :temperature 0 :max-tokens 1) :content))))))

(test local-backend-reentrant-session-is-an-error
  (with-local-backend (backend)
    (let ((nested nil))
      (ci:complete backend :messages *backend-messages* :temperature 0 :max-tokens 1
                           :stream (lambda (e)
                                     (when (eq :text-delta (getf e :type))
                                       (setf nested (ci:complete backend :messages *backend-messages*)))))
      (is (eq :bad-request (first (ci:result-error nested)))))))

(test local-backend-concurrent-session-policies
  (dolist (policy '(:serialized :per-call))
    (with-local-backend (backend :session-policy policy)
      (let* ((entered (bt2:make-semaphore))
             (release (bt2:make-semaphore))
             (first-result nil)
             (worker (bt2:make-thread
                      (lambda ()
                        (setf first-result
                              (ci:complete backend :messages *backend-messages* :temperature 0
                                                   :max-tokens 1
                                                   :stream (lambda (e)
                                                             (when (eq :text-delta (getf e :type))
                                                               (bt2:signal-semaphore entered)
                                                               (bt2:wait-on-semaphore release)))))))))
        (unwind-protect
             (progn
               (is (bt2:wait-on-semaphore entered :timeout 3))
               (let ((second (ci:complete backend :messages *backend-messages* :temperature 0
                                                  :max-tokens 1 :timeout 100)))
                 (if (eq policy :serialized)
                     (is (equal '(:error :timeout) second))
                     (is (equal "a" (ci:content-text (getf (second second) :content))))))
               (when (eq policy :serialized)
                 (let* ((token (ci:make-cancel-token))
                        (canceller (bt2:make-thread (lambda () (sleep 0.05) (ci:cancel token)))))
                   (unwind-protect
                        (is (equal '(:error :cancelled)
                                   (ci:complete backend :messages *backend-messages* :cancel token)))
                     (bt2:join-thread canceller)))))
          (bt2:signal-semaphore release)
          (bt2:join-thread worker))
        (is (not (ci:result-error-p first-result)))
        (is (not (ci:result-error-p (ci:complete backend :messages *backend-messages*
                                                         :temperature 0 :max-tokens 1))))))))

(defvar *backend-script* nil)

(defmethod local:sample :around ((sampler local:sampler) logits)
  (if *backend-script* (pop (car *backend-script*)) (call-next-method)))

(test local-backend-parses-generated-tools-and-reasoning
  (dolist (case `(("<|im_start|> multi_step_tool" local:qwen3-chat
                   ,(format nil "<think>~%private~%</think>~%~%ok<tool_call>{\"name\":\"weather\",\"arguments\":{\"city\":\"Paris\"}}</tool_call>")
                   "ok" "private" :weather)
                 ("<|start_header_id|> Cutting Knowledge Date" local:llama3-chat
                  "{\"name\":\"weather\",\"parameters\":{\"city\":\"Paris\"}}" "" nil :weather)
                 ("<|start_header_id|> Cutting Knowledge Date" local:llama3-chat
                  "<|python_tag|>print('é')" "" nil :python)))
    (destructuring-bind (source template output expected reasoning name) case
      (with-gguf-file (path (backend-fixture source))
        (local:with-weights (weights path)
          (let* ((backend (local:make-local-backend weights :n-ctx 2048))
                 (tokenizer (local::backend-tokenizer backend))
                 (script (append (coerce (local:encode tokenizer output :special t :add-bos nil) 'list)
                                 (list (if (eq template 'local:llama3-chat)
                                           (local:token-id tokenizer "<|eom_id|>") 268))))
                 (events nil)
                 (*backend-script* (list script))
                 (reply (ci:complete* backend :messages *backend-messages* :tools *parser-tools*
                                             :stream (lambda (e) (push e events)))))
            (is (equal expected (ci:content-text (getf reply :content))))
            (is (equal reasoning (getf (getf reply :meta) :reasoning)))
            (is (eq :tool-calls (getf (getf reply :meta) :finish-reason)))
            (is (eq name (getf (first (getf reply :tool-calls)) :name)))
            (is (equal expected (event-text (reverse events))))
            (is (= 1 (count :done events :key (lambda (e) (getf e :type)))))
            (is (null (car *backend-script*)))))))))

(test local-backend-does-not-swallow-parser-errors
  (with-gguf-file (path (backend-fixture "<|im_start|> multi_step_tool"))
    (local:with-weights (weights path)
      (let* ((backend (local:make-local-backend weights :n-ctx 2048))
             (*backend-script* (list (append (coerce (local:encode (local::backend-tokenizer backend)
                                                                  "<tool_call>{}</tool_call>"
                                                                  :add-bos nil) 'list) '(268))))
             (events nil)
             (result (ci:complete backend :messages *backend-messages* :tools *parser-tools*
                                          :stream (lambda (e) (push e events)))))
        (is (eq :backend-error (first (ci:result-error result))))
        (is (equal "" (event-text events)))
        (is (= 1 (length events)))
        (is (equal result (getf (first events) :reason)))
        (is (null (local::backend-session backend)))))))

(test local-backend-registration-and-signalling-sink
  (let ((original (ci:find-backend :local)))
    (unwind-protect
         (with-local-backend (backend :name :local)
           (ci:register-backend backend)
           (is (eq :ready (getf (ci:describe-backend (ci:find-backend :local)) :status)))
           (is (equal "a" (ci:content-text
                            (getf (ci:complete* :local :messages *backend-messages*
                                                      :temperature 0 :max-tokens 1
                                                      :stream (lambda (e)
                                                                (declare (ignore e))
                                                                (error "Sink failed")))
                                  :content)))))
      (ci:register-backend original))))

(test generation-special-decoding-is-opt-in
  (with-local-backend (backend)
    (let* ((tokenizer (local::backend-tokenizer backend))
           (tag (local:token-id tokenizer "<|python_tag|>"))
           (end (local:token-id tokenizer "<|eom_id|>")))
      (dolist (special '(nil t))
        (multiple-value-bind (text finish ids)
            (local:generate (local::backend-new-session backend) '(1) :special special
                            :sampler (script-sampler (list tag 97 end)) :stop-ids (list end))
          (is (equal (if special "<|python_tag|>a" "a") text))
          (is (eq :stop finish))
          (is (equal (list tag 97) ids)))))))

(test parser-finished-cannot-be-reused
  (let ((parser (local:make-reply-parser (make-instance 'local:chatml))))
    (local:reply-parser-push parser "hello")
    (is (equal "hello" (local:reply-parser-finish parser)))
    (signals local:reply-parser-error (local:reply-parser-finish parser))
    (signals local:reply-parser-error (local:reply-parser-push parser "more"))))

(test local-backend-cancel-during-decoding-and-recover
  (with-local-backend (backend)
    (let ((token (ci:make-cancel-token)) (events nil))
      (is (equal '(:error :cancelled)
                 (ci:complete backend :messages *backend-messages* :temperature 0 :max-tokens 10
                                      :cancel token
                                      :stream (lambda (e)
                                                (push e events)
                                                (when (eq :text-delta (getf e :type))
                                                  (ci:cancel token))))))
      (is (= 1 (count :done events :key (lambda (e) (getf e :type)))))
      (is (equal '(:error :cancelled) (getf (first events) :reason))))
    (is (equal "a" (ci:content-text
                     (getf (ci:complete* backend :messages *backend-messages*
                                                :temperature 0 :max-tokens 1) :content))))))

(test local-backend-text-blocks-and-unsupported-tool-history
  (with-local-backend (backend)
    (is (equal "a" (ci:content-text
                     (getf (ci:complete* backend :messages (list (list :role :user :content
                                                                      (ci:normalize-content "hi")))
                                                :temperature 0 :max-tokens 1) :content))))
    (is (eq :bad-request
            (first (ci:result-error
                    (ci:complete backend :messages '((:role :tool :tool-call-id "c1" :content "ok")))))))))
