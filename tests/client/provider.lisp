(in-package #:cl-inference/client/tests)
(in-suite :cl-inference/client)

;;; The provider layer against the fake HTTP backend: what a declaration
;;; publishes, what an instance may override, where a credential comes from and
;;; what it looks like on the wire, and that a turn crosses the extra hop
;;; without losing anything.

;;; Every declaration pins a base URL nothing listens on, because the fake
;;; backend's port is only known at run time: each instance overrides it, which
;;; is the same override a remote Ollama host or a proxy uses.

(client:define-provider :test-keyed
  :protocol :protocol-openai
  :base-url "http://127.0.0.1:1"
  :auth '(:bearer :env "CL_INFERENCE_TEST_KEY_NEVER_SET")
  :models '("test-model" "test-model-large")
  :defaults '(:temperature 0.25)
  :headers '("x-quirk" "on")
  :summary "A bearer-keyed provider for tests")

(client:define-provider :test-header-keyed
  :protocol :protocol-openai
  :base-url "http://127.0.0.1:1"
  :auth '(:header "x-api-key" :env "CL_INFERENCE_TEST_KEY_NEVER_SET"))

(client:define-provider :test-keyless
  :protocol :protocol-openai
  :base-url "http://127.0.0.1:1"
  :auth :none)

;;; PATH rather than a variable the test sets: no implementation offers a
;;; portable SETENV, and PATH is the one variable guaranteed to be there. What
;;; is under test is that the key is read from the variable the declaration
;;; names, not which variable that is.

(client:define-provider :test-env-keyed
  :protocol :protocol-openai
  :base-url "http://127.0.0.1:1"
  :auth '(:bearer :env "PATH"))

(client:register-backend (client:make-scripted-backend :name :protocol-echo :echo t))

(client:define-provider :test-echo
  :protocol :protocol-echo
  :base-url "http://127.0.0.1:1"
  :headers '("x-quirk" "on")
  :defaults '(:temperature 0.25))

(client:define-provider :test-rewriting
  :protocol :protocol-echo
  :base-url "http://127.0.0.1:1"
  :rewrite-request (lambda (request) (list* :rewritten-request t request))
  :rewrite-response (lambda (result)
                      (if (eq :ok (first result))
                          (list :ok (list* :rewritten t (second result)))
                          result)))

(client:define-provider :test-stacked
  :protocol :test-echo
  :base-url "http://127.0.0.1:2")

(client:define-provider :test-orphan
  :protocol :protocol-nobody-registered
  :base-url "http://127.0.0.1:1")

;;; --- the harness ------------------------------------------------------

(defun call-with-providers (answer body)
  (let ((server (start-fake-http
                 (lambda (&rest request)
                   (if (functionp answer) (apply answer request) answer)))))
    (setf *backend* server)
    (unwind-protect (funcall body)
      (stop-fake-http server))))

(defmacro with-providers ((answer) &body body)
  `(call-with-providers ,answer (lambda () ,@body)))

(defun instance (name &rest initargs)
  "NAME's provider pointed at the fake backend."
  (apply #'client:make-provider name :base-url (fake-http-url *backend*) initargs))

(defun keyed (&rest initargs)
  "The bearer-keyed provider. INITARGS come first, since the leftmost initarg
is the one that counts."
  (apply #'instance :test-keyed (append initargs '(:model "test-model" :api-key "sk-secret"))))

(defun turn (provider &rest extra)
  (apply #'client:complete provider :messages '((:role :user :content "hello")) extra))

(defun sent-header (name)
  (getf-string (getf (first (fake-http-requests *backend*)) :headers) name))

;;; --- the declaration --------------------------------------------------

(test a-provider-is-discoverable-and-describes-itself
  (with-providers ((json-response +hello-reply+))
    (is (member :test-keyed (client:providers)))
    (is (null (member :protocol-openai (client:providers))))
    (let ((metadata (client:describe-backend (keyed))))
      (is (eq :provider (getf metadata :kind)))
      (is (eq :test-keyed (getf metadata :name)))
      (is (eq :protocol-openai (getf metadata :protocol)))
      (is (stringp (getf metadata :summary)))
      (is (equal '("test-model" "test-model-large") (getf metadata :models)))
      (is (equal '(:temperature 0.25) (getf metadata :defaults)))
      (is (eq :ready (getf metadata :status)))
      ;; The auth form names its kind and where the key comes from, never
      ;; the key.
      (is (eq :bearer (getf (getf metadata :auth) :kind)))
      (is (equal "CL_INFERENCE_TEST_KEY_NEVER_SET" (getf (getf metadata :auth) :env))))))

(test metadata-carries-no-key-material
  (with-providers ((json-response +hello-reply+))
    (is (null (search "sk-secret" (princ-to-string (client:describe-backend (keyed))))))))

(test an-instance-overrides-the-declaration
  ;; The declaration pins a dead port; only the override makes the turn land.
  (with-providers ((json-response +hello-reply+))
    (let* ((provider (keyed :model "override-model"))
           (metadata (client:describe-backend provider)))
      (is (equal (fake-http-url *backend*) (getf metadata :base-url)))
      (is (equal "override-model" (getf metadata :model)))
      (is (eq :ok (first (turn provider))))
      (is (equal "override-model" (gethash "model" (sent-body)))))))

(test a-registered-provider-is-addressable-by-name
  (is (equal "http://127.0.0.1:1"
             (client:provider-base-url (client:find-backend :test-keyed)))))

(test only-a-defined-provider-can-be-instanced
  (signals error (client:make-provider :protocol-openai))
  (signals error (client:make-provider :never-defined)))

(test a-declaration-outside-the-vocabulary-is-a-definition-error
  (flet ((declaration (&rest plist)
           (signals error
             (client::check-provider-declaration
              :broken
              (append plist '(:protocol :protocol-openai
                              :base-url "http://127.0.0.1:1"))))))
    (declaration :auth :oauth)                     ; not one of the three kinds
    (declaration :auth '(:bearer))                 ; a bearer with no :env
    (declaration :auth '(:header :env "VAR"))      ; a header with no name
    (declaration :defaults '(:temperature))        ; not a plist
    (declaration :quirk t))                        ; not a key at all
  ;; And the two the declaration cannot do without.
  (signals error (client::check-provider-declaration :b '(:base-url "http://x")))
  (signals error (client::check-provider-declaration :b '(:protocol :protocol-openai)))
  (signals error (client::check-provider-declaration
                  :b '(:protocol :protocol-openai :base-url "127.0.0.1:11434"))))

;;; --- credentials ------------------------------------------------------

(test bearer-auth-reaches-the-wire
  (with-providers ((json-response +hello-reply+))
    (is (eq :ok (first (turn (keyed)))))
    (is (equal "Bearer sk-secret" (sent-header "authorization")))))

(test header-auth-reaches-the-wire-under-its-own-name
  (with-providers ((json-response +hello-reply+))
    (is (eq :ok (first (turn (instance :test-header-keyed
                                       :model "test-model" :api-key "sk-secret")))))
    (is (equal "sk-secret" (sent-header "x-api-key")))
    (is (null (sent-header "authorization")))))

(test a-keyless-provider-authenticates-with-nothing
  (with-providers ((json-response +hello-reply+))
    (let ((provider (instance :test-keyless :model "test-model")))
      (is (eq :ok (first (turn provider))))
      (is (null (sent-header "authorization")))
      (is (eq :none (getf (getf (client:describe-backend provider) :auth) :kind))))))

(test a-key-comes-from-the-environment-variable-the-declaration-names
  (with-providers ((json-response +hello-reply+))
    (is (eq :ok (first (turn (instance :test-env-keyed :model "test-model")))))
    (is (equal (format nil "Bearer ~a" (uiop:getenv "PATH"))
               (sent-header "authorization")))))

(test a-provider-with-no-key-is-unavailable-and-stays-off-the-wire
  ;; Discovery lists it and the reason is legible without a call; the call
  ;; itself is a bad request rather than an outage.
  (with-providers ((json-response +hello-reply+))
    (let ((provider (instance :test-keyed :model "test-model")))
      (is (eq :unavailable (getf (client:describe-backend provider) :status)))
      (let ((reason (client:result-error (turn provider))))
        (is (eq :bad-request (first reason)))
        (is (search "CL_INFERENCE_TEST_KEY_NEVER_SET" (second reason))))
      (is (null (fake-http-requests *backend*))))))

;;; --- layering ---------------------------------------------------------

(test defaults-layer-under-the-request
  (with-providers ((json-response +hello-reply+))
    (turn (keyed))
    (is (= 0.25d0 (gethash "temperature" (sent-body)))))
  (with-providers ((json-response +hello-reply+))
    (turn (keyed) :temperature 0.9)
    (is (= 0.9d0 (gethash "temperature" (sent-body))))))

(test quirk-headers-travel-and-the-caller-outranks-them
  (with-providers ((json-response +hello-reply+))
    (turn (keyed))
    (is (equal "on" (sent-header "x-quirk"))))
  (with-providers ((json-response +hello-reply+))
    ;; Matched without case, as HTTP names are.
    (turn (keyed) :headers '("X-Quirk" "off"))
    (is (equal "off" (sent-header "x-quirk")))))

(test an-instance-that-binds-no-model-leaves-the-requirement-to-the-caller
  (with-providers ((json-response +hello-reply+))
    (let ((provider (instance :test-keyless)))
      (is (eq :bad-request (first (client:result-error (turn provider)))))
      (is (eq :ok (first (turn provider :model "asked-for"))))
      (is (equal "asked-for" (gethash "model" (sent-body)))))))

(test the-provider-layers-its-data-under-the-request-it-hands-down
  (let* ((provider (client:make-provider :test-echo :base-url "http://example.test"
                                                    :model "m1"))
         (request (getf (getf (second (turn provider :top-p 0.5)) :meta) :request)))
    (is (equal "http://example.test" (getf request :base-url)))
    (is (equal "m1" (getf request :model)))
    (is (= 0.25 (getf request :temperature)))
    (is (= 0.5 (getf request :top-p)))
    (is (equal "on" (client::getf-ci (getf request :headers) "x-quirk")))))

;;; --- the turn crosses the hop -----------------------------------------

(test a-turn-crosses-the-provider-unchanged
  (with-providers ((json-response +hello-reply+))
    (let ((reply (second (turn (keyed)))))
      (is (equal "hi there" (client:content-text (getf reply :content))))
      (is (eq :stop (getf (getf reply :meta) :finish-reason)))
      (is (= 7 (getf (getf (getf reply :meta) :usage) :prompt-tokens))))))

(test a-streamed-turn-crosses-the-provider
  (with-providers ((sse-response
                    "{\"choices\":[{\"delta\":{\"content\":\"hi \"}}]}"
                    "{\"choices\":[{\"delta\":{\"content\":\"there\"}}]}"
                    "{\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}"
                    "[DONE]"))
    (let* ((events '())
           (result (turn (keyed) :ref :r1 :stream (lambda (event) (push event events)))))
      (is (equal '(:text-delta :text-delta :done)
                 (mapcar (lambda (event) (getf event :type)) (nreverse events))))
      (is (equal "hi there" (client:content-text (getf (second result) :content)))))))

(test a-cancel-crosses-the-provider
  (with-providers ((stalled-stream "text/event-stream"
                                   (sse-body "{\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}")))
    (with-hold
      (let* ((token (client:make-cancel-token))
             (canceller (bt:make-thread (lambda () (sleep 0.3) (client:cancel token))))
             (result (turn (keyed) :cancel token :timeout 30000
                                   :stream (lambda (event) event))))
        (bt:join-thread canceller)
        (is (eq :cancelled (client:result-error result)))))))

(test a-tool-call-crosses-the-provider-ready-to-invoke
  (with-providers ((json-response
                    "{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":null,
                       \"tool_calls\":[{\"id\":\"c9\",\"type\":\"function\",
                         \"function\":{\"name\":\"tool-shell\",
                           \"arguments\":\"{\\\"cmd\\\":\\\"ls\\\"}\"}}]},
                       \"finish_reason\":\"tool_calls\"}]}"))
    (let ((call (first (getf (second (turn (keyed))) :tool-calls))))
      (is (eq :tool-shell (getf call :name)))
      (is (equal "ls" (getf (getf call :arguments) :cmd))))))

;;; --- quirks -----------------------------------------------------------

(test a-provider-that-rewrites-the-request-and-response-still-does
  (let* ((result (turn (client:make-provider :test-rewriting))))
    (is (eq :ok (first result)))
    (is-true (getf (second result) :rewritten))
    (is-true (getf (getf (getf (second result) :meta) :request) :rewritten-request))))

(test a-provider-over-a-provider-completes
  (let ((result (turn (client:make-provider :test-stacked))))
    (is (eq :ok (first result)))
    ;; The inner provider's base URL wins over the outer's, as the caller's
    ;; data wins over a provider's.
    (is (equal "http://127.0.0.1:2"
               (getf (getf (getf (second result) :meta) :request) :base-url)))))

;;; --- errors -----------------------------------------------------------

(test a-backend-error-crosses-the-provider-intact
  (with-providers ('(429 ("Content-Type" "application/json")
                    "{\"error\":{\"message\":\"rate limited\"}}"))
    (let ((reason (client:result-error (turn (keyed)))))
      (is (eq :backend-error (first reason)))
      (is (= 429 (second reason)))
      (is (search "rate limited" (third reason))))))

(test an-unreachable-backend-crosses-the-provider-as-unavailable
  (with-providers (:close)
    (is (eq :unavailable (client:result-error (turn (keyed)))))))

(test a-malformed-request-never-leaves-the-provider
  (with-providers ((json-response +hello-reply+))
    (is (eq :bad-request
            (first (client:result-error (client:complete (keyed)
                                                         :messages '((:role :wizard)))))))
    (is (null (fake-http-requests *backend*)))))

(test a-provider-whose-protocol-is-not-registered-is-unavailable
  (is (eq :unavailable (client:result-error (turn (client:make-provider :test-orphan))))))

;;; --- the ollama provider ----------------------------------------------

(test ollama-declares-a-keyless-native-backend
  (with-providers ((json-response
                    "{\"message\":{\"role\":\"assistant\",\"content\":\"hi\"},
                      \"done\":true,\"done_reason\":\"stop\"}"))
    (let* ((provider (instance :ollama :model "llama3.2"))
           (metadata (client:describe-backend provider)))
      (is (eq :provider (getf metadata :kind)))
      (is (eq :protocol-ollama (getf metadata :protocol)))
      (is (eq :none (getf (getf metadata :auth) :kind)))
      (is (eq :ready (getf metadata :status)))
      (is (member "llama3.2" (getf metadata :models) :test #'equal))
      (is (eq :ok (first (turn provider))))
      (is (equal "llama3.2" (gethash "model" (sent-body))))
      (is (null (sent-header "authorization"))))))

(test ollama-pins-the-local-endpoint-by-default
  ;; The OpenAI-compatible /v1 route stays reachable with no provider of its
  ;; own -- :protocol-openai takes :base-url per request.
  (is (equal "http://127.0.0.1:11434"
             (client:provider-base-url (client:find-backend :ollama)))))

;;; --- live -------------------------------------------------------------

(test ollama-live-completion-through-the-provider
  ;; Off by default: CI must not depend on a model being installed. A
  ;; separate variable from CL_INFERENCE_OLLAMA_URL, which the OpenAI
  ;; protocol's live tests point at the /v1 route.
  (let ((base-url (uiop:getenv "CL_INFERENCE_OLLAMA_NATIVE_URL"))
        (model (or (uiop:getenv "CL_INFERENCE_OLLAMA_MODEL") "llama3.2")))
    (if (null base-url)
        (skip "set CL_INFERENCE_OLLAMA_NATIVE_URL to run live Ollama tests")
        (let ((result (client:complete
                       (client:make-provider :ollama :base-url base-url :model model)
                       :timeout 120000
                       :messages '((:role :user :content "Reply with the word ok.")))))
          (if (model-missing-p result)
              (skip "~a has no model ~a; set CL_INFERENCE_OLLAMA_MODEL" base-url model)
              (progn
                (is (eq :ok (first result)))
                (is (plusp (length (client:content-text (getf (second result) :content)))))))))))
