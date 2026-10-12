(in-package #:cl-inference/local)

;;; The in-process model, addressed through the client contract as :local.

(defclass local-backend (ci:backend) ())

(defmethod ci:describe-backend ((backend local-backend))
  (list :kind :provider :name (ci:backend-name backend)
        :summary "In-process model" :status :unavailable))

;; TODO: run the request through a session, sampler and chat template (#24)
(defmethod ci:backend-complete ((backend local-backend) request)
  (declare (ignore request))
  (ci:fail :unavailable))

(ci:register-backend (make-instance 'local-backend :name :local))
