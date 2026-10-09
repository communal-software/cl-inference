(in-package #:cl-inference/client/tests)
(in-suite :cl-inference/client)

;;; HTTP-only consumers must not load the engine.

(defun dependency-name (dependency)
  (typecase dependency
    (string dependency)
    (cons (case (first dependency)
            (:version (second dependency))
            (:feature (dependency-name (third dependency)))
            (:require (second dependency))))))

(defun all-dependencies (system-name)
  "Every system SYSTEM-NAME depends on, directly or not."
  (let ((seen '()))
    (labels ((visit (name)
               (unless (member name seen :test #'string-equal)
                 (push name seen)
                 (let ((system (asdf:find-system name nil)))
                   (when system
                     (dolist (dependency (asdf:system-depends-on system))
                       (a-visit (dependency-name dependency)))))))
             (a-visit (name) (when name (visit name))))
      (visit system-name))
    (rest (reverse seen))))

(test the-client-does-not-depend-on-the-engine
  (let ((dependencies (all-dependencies "cl-inference/client")))
    (is (plusp (length dependencies)))
    (is (null (remove-if-not (lambda (name)
                               (or (string-equal name "cl-inference")
                                   (string-equal name "cl-tensor")
                                   (and (> (length name) 12)
                                        (string-equal "cl-inference/" name :end2 13))))
                             dependencies)))))
