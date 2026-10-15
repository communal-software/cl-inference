(in-package #:cl-inference/examples)

(defun chat-file (path prompts &key (cache-type :f32) (max-tokens 64) (timeout 30000))
  "Return greedy replies for successive user PROMPTS, reusing one session's token prefix."
  (local:with-weights (weights path)
    (let ((backend (local:make-local-backend weights :n-ctx 2048 :cache-type cache-type))
          (messages nil))
      (unwind-protect
           (loop for prompt in prompts
                 for next = (append messages (list (list :role :user :content prompt)))
                 for reply = (ci:complete* backend :messages next :temperature 0
                                                   :max-tokens max-tokens :timeout timeout)
                 collect reply
                 do (setf messages (append next
                                           (list (list :role :assistant :content (getf reply :content))))))
        (local:close-local-backend backend)))))
