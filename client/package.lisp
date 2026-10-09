(defpackage #:cl-inference/client
  (:use #:cl)
  (:local-nicknames (#:a #:alexandria)
                    (#:json #:com.inuoe.jzon)
                    (#:bt #:bordeaux-threads-2))
  (:export
   ;; calling
   #:complete #:complete* #:check-request
   ;; results
   #:fail #:bad-request #:backend-error #:result-error-p #:result-error
   ;; conditions
   #:completion-error #:completion-error-reason #:completion-timeout #:completion-cancelled
   #:completion-unavailable #:completion-bad-request #:completion-backend-error
   #:completion-backend-error-status #:completion-backend-error-detail
   #:completion-backend-error-retry-after
   ;; content and events
   #:normalize-content #:content-text #:text-block
   #:text-delta #:tool-call-delta #:done #:emit-event
   ;; cancellation
   #:make-cancel-token #:cancel #:cancelled-p #:on-cancel
   ;; backends
   #:backend #:backend-complete #:describe-backend #:register-backend #:find-backend
   #:backends #:protocols #:protocol-backend #:openai-backend #:ollama-backend #:scripted-backend #:make-scripted-backend #:scripted-requests
   ;; providers
   #:define-provider #:provider #:make-provider #:providers
   #:provider-base-url #:provider-protocol #:provider-model #:provider-declaration
   ;; schema
   #:any #:array-of #:map-of #:object
   #:coerce-args #:validate-schema #:schema->json-schema #:json-schema->schema
   ;; transport
   #:+default-timeout+ #:*sink-grace*))
