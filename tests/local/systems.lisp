(in-package #:cl-inference/local/tests)

(in-suite :cl-inference/local)

(test packages-load
  (dolist (name '("CL-INFERENCE" "CL-INFERENCE/LOCAL" "CL-INFERENCE/GGUF" "CL-INFERENCE/QUANT"))
    (is (find-package name))))

(test the-unconfigured-local-backend-is-unavailable
  (is (member :local (ci:backends :kind :provider)))
  (is (eq :unavailable (getf (ci:describe-backend (ci:find-backend :local)) :status)))
  (is (equal '(:error :unavailable)
             (ci:complete :local :messages '((:role :user :content "hi"))))))
