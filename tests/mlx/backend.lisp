(in-package #:cl-inference/mlx/tests)
(in-suite :cl-inference/mlx)

(defun write-mlx-backend-fixture (directory)
  (let* ((spec (f::spec* :vocab 276 :qk-norm t)) (table (f::weight-table spec)))
    (maphash (lambda (name entry)
               (setf (cdr entry)
                     (loop for i below (length (cdr entry))
                           collect (if (or (string= name "token_embd.weight") (search "norm.weight" name)
                                           (and (string= name "output.weight") (= (floor i 8) 97))) 1f0 0f0)))) table)
    (f::write-hf-model directory "qwen3" spec table)
    (f::write-hf-tokenizer-fixture directory "qwen3")))

(defmacro with-backend ((backend &rest options) &body body)
  `(f::with-hf-directory (directory)
     (write-mlx-backend-fixture directory)
     (local:with-weights (weights directory)
       (mlx:with-mlx-backend (,backend weights ,@options) ,@body))))

(test mlx-client-contract-and-session-policies
  (dolist (policy '(:serialized :per-call))
    (let ((before (mx:live-array-count)))
      (with-backend (backend :session-policy policy :n-batch 5)
        (let* ((events nil)
               (reply (ci:complete* backend :messages f::*backend-messages* :temperature 0 :max-tokens 3
                                            :ref :turn :stream (lambda (e) (push e events)))))
          (is (equal "aaa" (ci:content-text (getf reply :content))))
          (is (eq :length (getf (getf reply :meta) :finish-reason)))
          (is (equal "aaa" (f::event-text (reverse events))))
          (is (= 1 (count :done events :key (lambda (e) (getf e :type))))))
        (let* ((request (list :messages f::*backend-messages* :temperature 0.6 :top-k 4 :top-p 0.9 :seed 42 :max-tokens 4))
               (a (apply #'ci:complete* backend request)) (b (apply #'ci:complete* backend request)))
          (is (equal (getf a :content) (getf b :content))))
        (is (equal "" (ci:content-text
                         (getf (ci:complete* backend :messages f::*backend-messages* :temperature 0 :stop '("aa") :max-tokens 8)
                               :content))))
        (is (eq :ready (getf (ci:describe-backend backend) :status))))
      (is (= before (mx:live-array-count))))))

(test mlx-deadline-cancel-and-recovery
  (with-backend (backend)
    (let ((cancel (ci:make-cancel-token)) (events nil))
      (ci:cancel cancel)
      (is (equal '(:error :cancelled)
                 (ci:complete backend :messages f::*backend-messages* :cancel cancel
                                      :stream (lambda (e) (push e events)))))
      (is (= 1 (count :done events :key (lambda (e) (getf e :type))))))
    (is (equal '(:error :timeout) (ci:complete backend :messages f::*backend-messages* :timeout 0)))
    (let ((cancel (ci:make-cancel-token)))
      (is (equal '(:error :cancelled)
                 (ci:complete backend :messages f::*backend-messages* :temperature 0 :max-tokens 10 :cancel cancel
                                      :stream (lambda (e) (when (eq :text-delta (getf e :type)) (ci:cancel cancel)))))))
    (is (equal "aa" (ci:content-text
                      (getf (ci:complete* backend :messages f::*backend-messages* :temperature 0 :max-tokens 2) :content))))))

(test mlx-q8-interruption-and-prefix-recovery
  (let ((before (mx:live-array-count)))
    (with-backend (backend :cache-type :q8)
      (ci:complete* backend :messages f::*backend-messages* :temperature 0 :max-tokens 2)
      (let ((cancel (ci:make-cancel-token)))
        (is (equal '(:error :cancelled)
                   (ci:complete backend :messages f::*backend-messages* :temperature 0 :max-tokens 8
                                        :cancel cancel
                                        :stream (lambda (event)
                                                  (when (eq :text-delta (getf event :type))
                                                    (ci:cancel cancel)))))))
      (is (null (local::backend-session backend)))
      (ci:complete* backend :messages f::*backend-messages* :temperature 0 :max-tokens 2)
      (let* ((reply (ci:complete* backend :messages f::*backend-messages* :temperature 0 :max-tokens 2))
             (usage (getf (getf reply :meta) :usage)))
        (is (equal "aa" (ci:content-text (getf reply :content))))
        (is (plusp (getf usage :cached-prompt-tokens)))))
    (is (= before (mx:live-array-count)))))

(test mlx-invalid-requests-and-lifetime
  (with-backend (backend :n-ctx 2)
    (is (eq :bad-request (first (ci:result-error (ci:complete backend :messages f::*backend-messages*)))))
    (is (eq :bad-request (first (ci:result-error (ci:complete backend :messages f::*backend-messages* :top-k 0)))))
    (mlx:close-mlx-backend backend)
    (is (eq :unavailable (getf (ci:describe-backend backend) :status)))
    (is (equal '(:error :unavailable) (ci:complete backend :messages f::*backend-messages*))))
  (with-backend (backend)
    (local:close-weights (local::backend-weights backend))
    (is (equal '(:error :unavailable) (ci:complete backend :messages f::*backend-messages*))))
  (is (equal '(:error :unavailable) (ci:complete :mlx :messages f::*backend-messages*))))

(test mlx-concurrent-session-policies
  (dolist (policy '(:serialized :per-call))
    (with-backend (backend :session-policy policy)
      (let ((results (make-array 3)))
        (let ((threads (loop for i below 3 collect
                             (let ((index i))
                               (bt:make-thread (lambda ()
                                                 (setf (aref results index)
                                                       (ci:complete backend :messages f::*backend-messages* :temperature 0 :max-tokens 2))))))))
          (mapc #'bt:join-thread threads))
        (is (every (lambda (result) (and (not (ci:result-error-p result))
                                         (equal "aa" (ci:content-text (getf (second result) :content))))) results))))))

(test mlx-tools-reasoning-and-parser-failure
  (with-backend (backend :n-ctx 2048)
    (let* ((tokenizer (local::backend-tokenizer backend))
           (output (format nil "<think>~%private~%</think>~%~%ok<tool_call>{\"name\":\"weather\",\"arguments\":{\"city\":\"Paris\"}}</tool_call>"))
           (f::*backend-script* (list (append (coerce (local:encode tokenizer output :special t) 'list) '(268))))
           (events nil)
           (reply (ci:complete* backend :messages f::*backend-messages* :tools f::*parser-tools*
                                       :stream (lambda (event) (push event events)))))
      (is (equal "ok" (ci:content-text (getf reply :content))))
      (is (equal "private" (getf (getf reply :meta) :reasoning)))
      (is (eq :weather (getf (first (getf reply :tool-calls)) :name)))
      (is (eq :tool-calls (getf (getf reply :meta) :finish-reason)))
      (is (= 1 (count :done events :key (lambda (e) (getf e :type))))))
    (let* ((f::*backend-script* (list (append (coerce (local:encode (local::backend-tokenizer backend)
                                                                  "<tool_call>{}</tool_call>" :special t) 'list) '(268))))
           (result (ci:complete backend :messages f::*backend-messages* :tools f::*parser-tools*)))
      (is (eq :backend-error (first (ci:result-error result))))
      (is (null (local::backend-session backend))))))
