(in-package #:cl-inference/client)

;;; A completion answers (:ok reply) or (:error reason). Reasons:
;;;   :timeout | :cancelled | :unavailable
;;;   (:bad-request message)
;;;   (:backend-error status detail [:retry-after ms])

(defconstant +default-timeout+ 30000
  "Milliseconds a completion gets when the caller names no deadline.")

(defun fail (reason) (list :error reason))

(defun bad-request (format &rest args)
  (fail (list :bad-request (apply #'format nil format args))))

(defun backend-error (status detail &key retry-after)
  "The backend was reached and the exchange broke down: a non-OK STATUS, a
malformed payload, a stream cut short. RETRY-AFTER, in milliseconds, is how
long the backend asked a caller to wait before trying again."
  (fail (list* :backend-error status detail
               (when retry-after (list :retry-after retry-after)))))

(defun result-error-p (result)
  (and (consp result) (eq (first result) :error)))

(defun result-error (result)
  (when (result-error-p result) (second result)))

;;; --- conditions -------------------------------------------------------

(define-condition completion-error (error)
  ((reason :initarg :reason :reader completion-error-reason))
  (:report (lambda (condition stream)
             (format stream "Completion failed: ~s" (completion-error-reason condition)))))

(define-condition completion-timeout (completion-error) ()
  (:default-initargs :reason :timeout))

(define-condition completion-cancelled (completion-error) ()
  (:default-initargs :reason :cancelled))

(define-condition completion-unavailable (completion-error) ()
  (:default-initargs :reason :unavailable))

(define-condition completion-bad-request (completion-error) ())

(define-condition completion-backend-error (completion-error)
  ((status :initarg :status :reader completion-backend-error-status)
   (detail :initarg :detail :reader completion-backend-error-detail)
   (retry-after :initarg :retry-after :initform nil
                :reader completion-backend-error-retry-after)))

(defun signal-completion-error (reason)
  (typecase reason
    ((eql :timeout) (error 'completion-timeout))
    ((eql :cancelled) (error 'completion-cancelled))
    ((eql :unavailable) (error 'completion-unavailable))
    (cons (case (first reason)
            (:bad-request (error 'completion-bad-request :reason reason))
            (:backend-error
             (destructuring-bind (status detail &key retry-after) (rest reason)
               (error 'completion-backend-error
                      :reason reason :status status :detail detail
                      :retry-after retry-after)))
            (t (error 'completion-error :reason reason))))
    (t (error 'completion-error :reason reason))))
