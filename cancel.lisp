(in-package #:cl-inference)

;;; A caller-held handle rather than a message: it reaches a completion through
;;; every layer without the caller knowing which backend holds it.

(defstruct (cancel-token (:constructor make-cancel-token ()))
  (lock (bt:make-lock)) cancelled actions)

(defun cancelled-p (token)
  (bt:with-lock-held ((cancel-token-lock token))
    (cancel-token-cancelled token)))

(defun on-cancel (token function)
  "Call FUNCTION when TOKEN is cancelled, at once if it already is.
Return an idempotent remover; an action already claimed by CANCEL may run."
  (let ((registration (list function)))
    (if (bt:with-lock-held ((cancel-token-lock token))
          (if (cancel-token-cancelled token)
              t
              (progn (push registration (cancel-token-actions token)) nil)))
        (progn (funcall function) (constantly nil))
        (lambda ()
          (bt:with-lock-held ((cancel-token-lock token))
            (setf (cancel-token-actions token)
                  (delete registration (cancel-token-actions token) :test #'eq)))
          nil))))

(defun cancel (token)
  "Cancel whatever TOKEN was passed to as :CANCEL. Idempotent; true the first
time."
  (let* ((first nil)
         (actions (bt:with-lock-held ((cancel-token-lock token))
                    (unless (cancel-token-cancelled token)
                      (setf first t
                            (cancel-token-cancelled token) t)
                      (shiftf (cancel-token-actions token) nil)))))
    (dolist (registration actions) (funcall (car registration)))
    first))
