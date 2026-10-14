(in-package #:cl-inference/tests)
(in-suite :cl-inference)

;;; The shared transport: result shapes, Retry-After, cancel tokens and the
;;; exchange deadline.

(test a-null-sink-drops-events
  (is (equal '(:type :done :ref nil :reason nil)
             (client:emit-event nil (client:done nil)))))

(test a-sink-that-signals-does-not-fail-the-emit
  (is (equal '(:type :done :ref nil :reason nil)
             (client:emit-event (lambda (event) (declare (ignore event)) (error "boom"))
                                (client:done nil)))))

(test backend-error-is-its-own-shape
  (let ((result (client:backend-error 429 "rate limited")))
    (is (client:result-error-p result))
    (is (equal '(:backend-error 429 "rate limited") (client:result-error result)))))

(test backend-error-carries-a-retry-after-only-when-given
  (is (equal '(:backend-error 429 "slow" :retry-after 2000)
             (client:result-error (client:backend-error 429 "slow" :retry-after 2000)))))

(test retry-after-ms-reads-the-headers
  (flet ((wait (&rest headers) (client::retry-after-ms headers)))
    (is (= 2000 (wait '(:retry-after . "2"))))
    (is (= 1500 (wait '(:retry-after . "1.5"))))
    (is (= 1500 (wait '(:retry-after-ms . "1500"))))
    (is (= 250 (wait '(:retry-after-ms . "250") '(:retry-after . "9"))))
    (is (null (wait)))
    (is (null (wait '(:retry-after . "soon"))))
    (is (null (wait '(:retry-after . "-3"))))
    (is (null (wait '(:retry-after . ""))))
    (is (= 0 (wait '(:retry-after . "Wed, 21 Oct 2015 07:28:00 GMT"))))))

(test retry-after-ms-reads-an-http-date
  (let ((date (client::http-date-universal-time "Wed, 21 Oct 2026 07:28:00 GMT")))
    (is (= (encode-universal-time 0 28 7 21 10 2026 0) date))
    (is (null (client::http-date-universal-time "21 Oct 2026")))
    (is (null (client::http-date-universal-time "Wed, 21 Foo 2026 07:28:00 GMT")))
    (is (null (client::http-date-universal-time "Wed, 21 Oct 2026 07:28:00 PST")))))

(test tool-call-deltas-carry-argument-fragments
  (let ((event (client:tool-call-delta :r :id "c1" :name :tool-shell
                                          :arguments "{\"cmd\"")))
    (is (eq :tool-call-delta (getf event :type)))
    (is (equal "c1" (getf event :id)))
    (is (equal "{\"cmd\"" (getf event :arguments)))))

;;; --- cancel tokens --------------------------------------------------------

(defmacro with-replaced-function ((name function) &body body)
  (let ((original (gensym "ORIGINAL")))
    `(let ((,original (fdefinition ',name)))
       (unwind-protect
            (progn (setf (fdefinition ',name) ,function) ,@body)
         (setf (fdefinition ',name) ,original)))))

(test cancellation-during-registration-prevents-the-bounded-function
  (let ((token (client:make-cancel-token))
        (register (fdefinition 'client:on-cancel))
        (ran nil))
    (with-replaced-function
        (client:on-cancel (lambda (token action)
                            (client:cancel token)
                            (funcall register token action)))
      (is (equal '(nil :cancelled)
                 (multiple-value-list
                  (client::call-bounded 30 (lambda (exchange)
                                            (declare (ignore exchange))
                                            (setf ran t))
                                        :cancel token)))))
    (is-false ran)
    (is (null (client::cancel-token-actions token)))))

(test a-cancel-token-runs-its-actions-once
  (let ((token (client:make-cancel-token))
        (runs 0))
    (client:on-cancel token (lambda () (incf runs)))
    (is-false (client:cancelled-p token))
    (is-true (client:cancel token))
    (is-false (client:cancel token))
    (is-true (client:cancelled-p token))
    (is (= 1 runs))))

(test cancel-answers-true-the-first-time-with-no-actions
  (let ((token (client:make-cancel-token)))
    (is-true (client:cancel token))
    (is-false (client:cancel token))))

(test an-action-registered-after-cancel-runs-at-once
  (let ((token (client:make-cancel-token))
        (runs 0))
    (client:cancel token)
    (client:on-cancel token (lambda () (incf runs)))
    (is (= 1 runs))))

(test removing-a-cancel-action-is-idempotent-and-specific
  (let* ((token (client:make-cancel-token))
         (runs 0)
         (action (lambda () (incf runs)))
         (remove-action (client:on-cancel token action)))
    (client:on-cancel token action)
    (funcall remove-action)
    (funcall remove-action)
    (is (= 1 (length (client::cancel-token-actions token))))
    (client:cancel token)
    (funcall remove-action)
    (is (= 1 runs))))

(test a-cancelled-token-returns-a-no-op-remover
  (let ((token (client:make-cancel-token)) (runs 0))
    (client:cancel token)
    (let ((remove-action (client:on-cancel token (lambda () (incf runs)))))
      (funcall remove-action)
      (funcall remove-action))
    (is (= 1 runs))
    (is (null (client::cancel-token-actions token)))))

(test removal-does-not-suppress-an-action-already-claimed-by-cancel
  (let* ((token (client:make-cancel-token))
         (claimed (bt:make-semaphore))
         (resume (bt:make-semaphore))
         (runs 0)
         (remove-action (client:on-cancel token (lambda () (incf runs)))))
    (client:on-cancel token (lambda ()
                             (bt:signal-semaphore claimed)
                             (bt:wait-on-semaphore resume :timeout 2)))
    (let ((thread (bt:make-thread (lambda () (client:cancel token)))))
      (unwind-protect
           (progn
             (is-true (bt:wait-on-semaphore claimed :timeout 2))
             (funcall remove-action))
        (bt:signal-semaphore resume)
        (bt:join-thread thread)))
    (is (= 1 runs))))

(test bounded-runs-remove-their-cancel-actions-on-every-exit
  (let ((token (client:make-cancel-token)))
    (dotimes (i 100)
      (client::call-bounded 30 (lambda (exchange) (declare (ignore exchange)) :ok)
                            :cancel token))
    (is (null (client::cancel-token-actions token)))
    (signals error
      (client::call-bounded 30 (lambda (exchange) (declare (ignore exchange)) (error "backend"))
                            :cancel token))
    (is (null (client::cancel-token-actions token)))
    (is (equal '(nil :timeout)
               (multiple-value-list
                (client::call-bounded 0 (lambda (exchange) (declare (ignore exchange)) (sleep 2))
                                      :cancel token))))
    (is (null (client::cancel-token-actions token)))
    (is (equal '(nil :cancelled)
               (multiple-value-list
                (client::call-bounded 30 (lambda (exchange)
                                          (declare (ignore exchange))
                                          (client:cancel token))
                                      :cancel token))))
    (is (null (client::cancel-token-actions token)))))

;;; --- the exchange's deadline -------------------------------------------------

(test an-exchange-runs-on-the-calling-thread
  (is (equal (list (bt:current-thread) nil)
             (multiple-value-list
              (client::call-with-deadline 1000 (lambda (connect)
                                                 (declare (ignore connect))
                                                 (bt:current-thread)))))))

(test the-deadline-unwinds-an-exchange-that-does-not-return
  (let ((start (get-internal-real-time)))
    (is (equal '(nil :timeout)
               (multiple-value-list
                (client::call-with-deadline 200 (lambda (connect)
                                                  (declare (ignore connect))
                                                  (sleep 10))))))
    (is (< (elapsed-since start) 2))))

(test a-cancel-unwinds-an-exchange-that-does-not-return
  (let ((token (client:make-cancel-token))
        (start (get-internal-real-time)))
    (bt:make-thread (lambda () (sleep 0.2) (client:cancel token)))
    (is (equal '(nil :cancelled)
               (multiple-value-list
                (client::call-with-deadline 10000 (lambda (connect)
                                                    (declare (ignore connect))
                                                    (sleep 10))
                                            :cancel token))))
    (is (< (elapsed-since start) 2))))

(test a-finished-exchange-ignores-a-later-cancel-and-deadline
  (let ((token (client:make-cancel-token)))
    (is (equal '(:done nil)
               (multiple-value-list
                (client::call-with-deadline 300 (lambda (connect)
                                                  (declare (ignore connect))
                                                  :done)
                                            :cancel token))))
    (client:cancel token)
    ;; Past the deadline too: neither may reach this thread.
    (sleep 0.5)
    (is (equal '(:next nil)
               (multiple-value-list
                (client::call-with-deadline 1000 (lambda (connect)
                                                   (declare (ignore connect))
                                                   (sleep 0.3)
                                                   :next)))))))

(test a-request-cancelled-beforehand-never-starts
  (let ((token (client:make-cancel-token))
        (ran nil))
    (client:cancel token)
    (is (equal '(nil :cancelled)
               (multiple-value-list
                (client::call-with-deadline 1000 (lambda (connect)
                                                   (declare (ignore connect))
                                                   (setf ran t))
                                            :cancel token))))
    (is-false ran)))

(test a-bounded-run-leaves-no-watcher-behind
  (client::call-bounded 30 (lambda (exchange) (declare (ignore exchange)) :quick))
  (is-true (eventually #'watchers-idle-p)))

(defclass transport-test-socket (usocket:stream-usocket)
  ((closes :initform 0 :accessor test-socket-closes)
   (reads :initform 0 :accessor test-socket-reads))
  (:default-initargs :stream (flexi-streams:make-in-memory-input-stream #())))

(defmethod usocket:socket-stream :before ((socket transport-test-socket))
  (incf (test-socket-reads socket)))

(defmethod usocket:socket-shutdown ((socket transport-test-socket) direction)
  (declare (ignore direction)))

(defmethod usocket:socket-close ((socket transport-test-socket))
  (incf (test-socket-closes socket))
  (close (slot-value socket 'usocket::stream)))

(test an-acquired-socket-is-owned-before-cancel-or-timeout-unwinds
  (dolist (reason '(:cancelled :timeout))
    (let ((socket (make-instance 'transport-test-socket))
          (delivered (bt:make-semaphore)) (interrupt nil))
      (with-replaced-function
          (usocket:socket-connect
           (lambda (&rest args)
             (declare (ignore args))
             (let ((exchange client::*exchange*))
               (is (null (client::exchange-socket exchange)))
               (setf interrupt
                     (bt:make-thread
                      (lambda ()
                        (client::abandon-exchange exchange reason)
                        (bt:interrupt-thread
                         (client::exchange-thread exchange)
                         (lambda () (bt:signal-semaphore delivered))))))
               (is-true (eventually (lambda ()
                                     (bt:wait-on-semaphore delivered :timeout 0))))
               socket)))
        (unwind-protect
             (is (equal (list nil reason)
                        (multiple-value-list
                         (client::call-with-deadline 30000
                                                    (lambda (connect)
                                                      (funcall connect "http://localhost"))))))
          (when interrupt (bt:join-thread interrupt))))
      (is (= 1 (test-socket-closes socket)))
      (is (= 0 (test-socket-reads socket)))))
  (is-true (eventually #'watchers-idle-p)))

(test cancellation-during-a-failed-connect-preserves-its-reason
  (with-replaced-function
      (usocket:socket-connect (lambda (&rest args)
                               (declare (ignore args))
                               (client::abandon-exchange client::*exchange* :cancelled)
                               (error "connect failed")))
    (is (equal '(nil :cancelled)
               (multiple-value-list
                (client::call-with-deadline 30000
                                           (lambda (connect)
                                             (funcall connect "http://localhost"))))))))

(test socket-cleanup-covers-connect-and-stream-wrapping-exits
  (dolist (failure '(nil :connect :wrap))
    (let ((socket (make-instance 'transport-test-socket)))
      (with-replaced-function
          (usocket:socket-connect (lambda (&rest args)
                                   (declare (ignore args))
                                   (when (eq failure :connect) (error "connect failed"))
                                   socket))
        (with-replaced-function
            (cl+ssl:make-ssl-client-stream (lambda (&rest args)
                                           (declare (ignore args))
                                           (error "TLS wrapping failed")))
          (flet ((connect-stream ()
                   (client::call-with-deadline
                    30000 (lambda (connect)
                            (let ((stream (funcall connect (if (eq failure :wrap)
                                                              "https://localhost"
                                                              "http://localhost"))))
                              (is (typep stream 'flexi-streams:flexi-stream))
                              :stream)))))
            (if failure
                (signals error (connect-stream))
                (is (equal '(:stream nil) (multiple-value-list (connect-stream))))))))
      (is (= (if (eq failure :connect) 0 1) (test-socket-closes socket)))))
  (is-true (eventually #'watchers-idle-p)))
