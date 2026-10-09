(defpackage #:cl-inference/client/tests
  (:use #:cl #:fiveam)
  (:local-nicknames (#:client #:cl-inference/client) (#:bt #:bordeaux-threads-2))
  (:export #:run-tests))

(in-package #:cl-inference/client/tests)

(def-suite :cl-inference/client)

(defun run-tests ()
  (run! :cl-inference/client))

(defun eventually (function &optional (timeout 2))
  "Poll FUNCTION until it returns true or TIMEOUT seconds pass."
  (loop with deadline = (+ (get-internal-real-time) (* timeout internal-time-units-per-second))
        for value = (funcall function)
        until (or value (> (get-internal-real-time) deadline))
        do (sleep 0.01)
        finally (return value)))

(defun elapsed-since (started)
  "Seconds since STARTED, an internal real time."
  (/ (- (get-internal-real-time) started) internal-time-units-per-second))

(defun watchers-idle-p ()
  "True when no deadline watcher thread is left running."
  (notany (lambda (thread) (equal "completion deadline" (bt:thread-name thread)))
          (bt:all-threads)))
