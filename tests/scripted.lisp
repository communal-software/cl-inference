(in-package #:cl-inference/tests)
(in-suite :cl-inference)

;;; The contract through COMPLETE and COMPLETE*, against scripted backends that
;;; implement it and nothing else.

(defun hello (&rest extra)
  (append (list :messages '((:role :user :content "hello"))) extra))

(defun bad-request-p (result)
  (let ((reason (client:result-error result)))
    (and (consp reason) (eq :bad-request (first reason)))))

(test complete-performs-one-turn
  (let* ((backend (client:make-scripted-backend :echo t))
         (result (apply #'client:complete backend (hello))))
    (is (eq :ok (first result)))
    (let ((reply (second result)))
      (is (eq :assistant (getf reply :role)))
      (is (eq t (getf reply :done)))
      (is (equal "hello!" (client:content-text (getf reply :content))))
      (is (= 1 (getf (getf reply :meta) :echoed))))))

(test a-backend-is-addressed-by-name-or-by-object
  (let ((backend (client:make-scripted-backend :name :test-scripted :echo t)))
    (client:register-backend backend)
    (is (equal (apply #'client:complete backend (hello))
               (apply #'client:complete :test-scripted (hello))))
    (is (member :test-scripted (client:backends)))))

(test an-unknown-backend-name-is-unavailable
  (is (eq :unavailable (client:result-error (apply #'client:complete :nobody (hello))))))

(test unknown-request-keys-are-ignored
  ;; A portable caller may offer a superset: a key the backend does not know
  ;; must pass through rather than being rejected.
  (is (eq :ok (first (apply #'client:complete (client:make-scripted-backend :echo t)
                            (hello :top-k 40 :seed 7))))))

(test content-blocks-and-flat-strings-agree
  (let ((backend (client:make-scripted-backend :echo t)))
    (flet ((content (message)
             (getf (second (client:complete backend :messages (list message))) :content)))
      (is (equal (content '(:role :user :content "hello"))
                 (content '(:role :user :content ((:type :text :text "hello")))))))))

(test a-malformed-request-never-reaches-the-backend
  (let ((backend (client:make-scripted-backend :echo t)))
    (is (bad-request-p (client:complete backend)))
    (is (bad-request-p (client:complete backend :messages '())))
    (is (bad-request-p (client:complete backend :messages '((:role :bard :content "x")))))
    (is (bad-request-p (client:complete backend :messages '((:role :tool :content "x")))))
    (is (bad-request-p
         (client:complete backend
                          :messages '((:role :assistant :tool-calls ((:name :tool-shell)))))))
    (is (bad-request-p (apply #'client:complete backend (hello :stream "not a function"))))
    (is (null (client:scripted-requests backend)))))

(test the-four-roles-are-accepted
  (is (eq :ok (first (client:complete
                      (client:make-scripted-backend :echo t)
                      :messages '((:role :system :content "be terse")
                                  (:role :user :content "ls")
                                  (:role :assistant :content nil
                                   :tool-calls ((:id "c1" :name :tool-shell
                                                 :arguments (:cmd "ls"))))
                                  (:role :tool :tool-call-id "c1" :content "a.lisp")))))))

(test a-script-answers-in-order-then-runs-out
  (let ((backend (client:make-scripted-backend
                  :replies (list "one" (client:backend-error 500 "boom")
                                 (lambda (request) (format nil "~d" (length (getf request :messages))))))))
    (is (equal "one" (client:content-text (getf (second (apply #'client:complete backend (hello))) :content))))
    (is (equal '(:backend-error 500 "boom")
               (client:result-error (apply #'client:complete backend (hello)))))
    (is (equal "1" (client:content-text (getf (second (apply #'client:complete backend (hello))) :content))))
    (is (eq :unavailable (client:result-error (apply #'client:complete backend (hello)))))
    (is (= 4 (length (client:scripted-requests backend))))))

(test a-scripted-turn-streams-its-text-and-one-done
  (let* ((events '())
         (result (apply #'client:complete (client:make-scripted-backend :replies '("hi"))
                        (hello :ref :r1 :stream (lambda (event) (push event events))))))
    (is (eq :ok (first result)))
    (is (equal '(:text-delta :done) (mapcar (lambda (event) (getf event :type))
                                            (reverse events))))
    (is (eq :stop (getf (first events) :reason)))))

(test a-failed-scripted-turn-ends-in-one-failed-done
  (let ((events '()))
    (apply #'client:complete (client:make-scripted-backend)
           (hello :stream (lambda (event) (push event events))))
    (is (equal '((:type :done :ref nil :reason (:error :unavailable))) events))))

;;; --- complete* ----------------------------------------------------------

(test complete*-answers-the-reply
  (let ((reply (apply #'client:complete* (client:make-scripted-backend :replies '("ok")) (hello))))
    (is (eq :assistant (getf reply :role)))
    (is (equal "ok" (client:content-text (getf reply :content))))))

(test complete*-signals-the-matching-condition
  (flet ((signalled (reply)
           (handler-case (apply #'client:complete* (client:make-scripted-backend :replies (list reply)) (hello))
             (client:completion-error (e) e))))
    (is (typep (signalled (client:fail :timeout)) 'client:completion-timeout))
    (is (typep (signalled (client:fail :cancelled)) 'client:completion-cancelled))
    (is (typep (signalled (client:fail :unavailable)) 'client:completion-unavailable))
    (is (typep (signalled (client:bad-request "no ~a" "model")) 'client:completion-bad-request))
    (let ((e (signalled (client:backend-error 429 "slow down" :retry-after 2000))))
      (is (typep e 'client:completion-backend-error))
      (is (= 429 (client:completion-backend-error-status e)))
      (is (equal "slow down" (client:completion-backend-error-detail e)))
      (is (= 2000 (client:completion-backend-error-retry-after e))))
    (is (typep (signalled (client:fail '(:error "odd"))) 'client:completion-error))))

(test complete*-signals-a-bad-request-before-the-backend
  (let ((backend (client:make-scripted-backend :replies '("never"))))
    (signals client:completion-bad-request (client:complete* backend))
    (is (null (client:scripted-requests backend)))))

(test preflight-failures-close-the-stream-with-the-original-result
  (dolist (unknown '(nil t))
    (dolist (sink-mode '(:collect :signal :block :none))
      (let* ((backend (client:make-scripted-backend :echo t))
             (token (client:make-cancel-token))
             (thread (bt:current-thread))
             (client::*sink-grace* 0.05)
             (events nil)
             (unwound nil)
             (started (get-internal-real-time)))
        (client:cancel token)
        (let ((result
                (client:complete
                 (if unknown :nobody backend)
                 :messages '((:role :tool :content "x"))
                 :ref :preflight :cancel token :timeout 0
                 :stream (unless (eq sink-mode :none)
                           (lambda (event)
                             (is (eq thread (bt:current-thread)))
                             (push event events)
                             (case sink-mode
                               (:signal (error "sink failure"))
                               (:block (unwind-protect (sleep 10)
                                         (setf unwound t)))))))))
          (if unknown
              (is (equal '(:error :unavailable) result))
              (is (bad-request-p result)))
          (unless (eq sink-mode :none)
            (is (equal (list (client:done :preflight result)) events)))
          (when (eq sink-mode :block)
            (is (eq t unwound))
            (is (< (/ (- (get-internal-real-time) started)
                      internal-time-units-per-second) 2)))
          (is (null (client:scripted-requests backend))))))))

(test unknown-backend-with-a-valid-request-closes-the-stream
  (let (events)
    (let ((result (apply #'client:complete :nobody
                         (hello :stream (lambda (event) (push event events))))))
      (is (equal '(:error :unavailable) result))
      (is (equal (list (client:done nil result)) events)))))

(test complete*-closes-preflight-streams-before-signalling
  (let (events)
    (handler-case
        (client:complete* (client:make-scripted-backend)
                          :stream (lambda (event) (push event events)))
      (client:completion-bad-request ()
        (is (= 1 (length events)))
        (is (eq :done (getf (first events) :type)))
        (is (bad-request-p (getf (first events) :reason)))))))
