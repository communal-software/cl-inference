(in-package #:cl-inference/client/tests)
(in-suite :cl-inference/client)

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
