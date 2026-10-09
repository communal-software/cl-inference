(in-package #:cl-inference/client)

;;; A backend that answers from a script, for tests that must not reach a
;;; network. Each call takes the next reply:
;;;   a string        a text reply
;;;   (:ok ...) / (:error ...)  that result as it stands
;;;   a function      called with the request, answering either of the above
;;; Once the script runs out, a call is (:error :unavailable). With :ECHO, every
;;; call answers the last message's text and the request it received, whatever
;;; the script holds.

(defclass scripted-backend (protocol-backend)
  ((replies :initarg :replies :initform nil :accessor scripted-replies)
   (echo :initarg :echo :initform nil :reader scripted-echo)
   (requests :initform '() :accessor %scripted-requests)
   (lock :initform (bt:make-lock) :reader scripted-lock))
  (:default-initargs :summary "Scripted replies for offline tests"))

(defun make-scripted-backend (&key name replies echo)
  (make-instance 'scripted-backend :name name :replies replies :echo echo))

(defun scripted-requests (backend)
  "The requests BACKEND has received, oldest first."
  (bt:with-lock-held ((scripted-lock backend))
    (reverse (%scripted-requests backend))))

(defun scripted-result (backend request)
  (bt:with-lock-held ((scripted-lock backend))
    (push request (%scripted-requests backend))
    (cond
      ((scripted-echo backend) (echo-reply request))
      ((null (scripted-replies backend)) (fail :unavailable))
      (t (pop (scripted-replies backend))))))

(defun echo-reply (request)
  (let ((text (content-text (getf (car (last (getf request :messages))) :content))))
    (make-reply (concatenate 'string text "!") nil :stop
                (list :echoed (length (getf request :messages)) :request request))))

(defun text-reply (text)
  (make-reply text nil :stop nil))

(defmethod backend-complete ((backend scripted-backend) request)
  (let* ((cancel (getf request :cancel))
         (result (if (and cancel (cancelled-p cancel))
                     (fail :cancelled)
                     (let ((reply (scripted-result backend request)))
                       (when (functionp reply)
                         (setf reply (funcall reply request)))
                       (if (stringp reply) (text-reply reply) reply))))
         (sink (getf request :stream))
         (ref (getf request :ref)))
    (when sink
      (unless (result-error-p result)
        (a:when-let ((text (content-text (getf (second result) :content))))
          (when (plusp (length text))
            (emit-event sink (text-delta ref text)))))
      (emit-event sink (done ref (done-reason result))))
    result))
