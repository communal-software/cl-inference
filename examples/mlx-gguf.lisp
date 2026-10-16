(require :asdf)
(let ((setup (merge-pathnames "quicklisp/setup.lisp" (user-homedir-pathname))))
  (when (and (not (find-package :ql)) (probe-file setup)) (load setup)))
(asdf:initialize-source-registry
 `(:source-registry (:directory ,(merge-pathnames "../" (uiop:pathname-directory-pathname *load-truename*)))
                   :inherit-configuration))
(asdf:load-system "cl-inference/mlx-gguf")

(destructuring-bind (path &optional (prompt "Say hello in one sentence."))
    (remove "--" (uiop:command-line-arguments) :test #'equal :count 1)
  (cl-inference/local:with-weights (weights path)
    (cl-inference/mlx:with-mlx-backend (backend weights :n-ctx 2048)
      (cl-inference:complete* backend :messages `((:role :user :content ,prompt))
        :temperature 0 :max-tokens 64 :enable-thinking nil
        :stream (lambda (event)
                  (when (eq (getf event :type) :text-delta)
                    (write-string (getf event :text)) (finish-output)))))))
