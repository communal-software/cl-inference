(in-package #:cl-inference/local/tests)

(in-suite :cl-inference/local)

(local:define-architecture ses-llama (blk-llama)
  (:format :gguf (:arch "ses-llama")))

(defun greedy-continuation (model tokens count)
  "COUNT greedy ids after TOKENS, each from a full pass with no cache."
  (let ((tokens (copy-list tokens)) (new nil))
    (dotimes (i count (nreverse new))
      (let ((id (local:greedy-sampler (local:model-forward model tokens))))
        (push id new)
        (setf tokens (append tokens (list id)))))))

(defun script-sampler (ids)
  (lambda (logits)
    (declare (ignore logits))
    (pop ids)))

(defmacro with-session ((session model &rest options) (&rest spec-overrides) &body body)
  `(let ((spec (spec* ,@spec-overrides)))
     (with-spec-model (,model "blk-llama" spec (weight-table spec))
       (let ((,session (local:make-session ,model ,@options)))
         ,model
         ,@body))))

(defun fixture-tokenizer ()
  (with-gguf-file (path (tokenizer-file))
    (local:with-weights (weights path) (local:load-tokenizer weights))))

(defun piece (tokenizer text)
  (aref (enc tokenizer text) 0))

;;; Prefill

(test session-prefill-matches-the-reference
  (let* ((spec (spec*)) (table (weight-table spec)) (tokens '(3 0 5 2 6)))
    (dolist (n-batch '(1 2 512))
      (with-spec-model (model "blk-llama" spec table)
        (let ((session (local:make-session model :n-batch n-batch)))
          (is (rows-close-p (tensor-rows (local:session-prefill session tokens))
                            (last (ref-logits table spec :adjacent tokens)))
              "n-batch ~D" n-batch)
          (is (= 5 (local:session-position session)))
          (is (rows-close-p (tensor-rows (local:session-prefill session '(1)))
                            (last (ref-logits table spec :adjacent (append tokens '(1)))))
              "n-batch ~D" n-batch))))))

(test session-prefill-stops-when-the-check-fails
  (with-session (session model :n-batch 2) ()
    (let ((calls 0))
      (multiple-value-bind (logits failure)
          (local:session-prefill session '(1 2 3 4 5)
                                 :check (lambda () (when (= 2 (incf calls)) (ci:fail :timeout))))
        (is (null logits))
        (is (equal '(:error :timeout) failure))
        (is (= 2 (local:session-position session)))))))

(test session-reset-forgets-stale-rows
  (with-session (session model) ()
    (local:generate session '(3 0 5) :max-tokens 3)
    (local:session-reset session)
    (is (= 0 (local:session-position session)))
    (is (equal (greedy-continuation model '(1 2) 3)
               (nth-value 2 (local:generate session '(1 2) :max-tokens 3))))))

;;; Context size

(defmacro with-context-model ((model arch length) &body body)
  `(let* ((spec (spec*)) (table (weight-table spec)))
     (with-gguf-file (path (spec-gguf ,arch spec table
                                      ,(when length `(list (list "ses-llama.context_length" 4 ,length)))))
       (local:with-weights (weights path)
         (let ((,model (local:load-model weights)))
           ,@body)))))

(test n-ctx-defaults-to-the-models-context-capped-at-4096
  (with-context-model (model "ses-llama" 100)
    (is (= 100 (local:session-n-ctx (local:make-session model)))))
  (with-context-model (model "ses-llama" 131072)
    (is (= 4096 (local:session-n-ctx (local:make-session model)))))
  (with-context-model (model "ses-llama" nil)
    (is (= 4096 (local:session-n-ctx (local:make-session model)))))
  (with-context-model (model "blk-llama" nil)
    (is (= 4096 (local:session-n-ctx (local:make-session model))))
    (is (= 7 (local:session-n-ctx (local:make-session model :n-ctx 7))))))

;;; Generating

(test generate-greedy-matches-an-uncached-loop
  (with-session (session model) ()
    (multiple-value-bind (text finish ids usage) (local:generate session '(3 0 5) :max-tokens 4)
      (is (null text))
      (is (eq :length finish))
      (is (equal (greedy-continuation model '(3 0 5) 4) ids))
      (is (equal '(:prompt-tokens 3 :completion-tokens 4 :total-tokens 7) usage))
      (is (= 7 (local:session-position session)))
      (is (equal (greedy-continuation model (append '(3 0 5) ids '(4)) 2)
                 (nth-value 2 (local:generate session '(4) :max-tokens 2)))
          "a turn continues where the last one left the cache"))))

(test generate-finishes-on-a-stop-id
  (with-session (session model) ()
    (let ((first (first (greedy-continuation model '(3 0 5) 1))))
      (multiple-value-bind (text finish ids) (local:generate session '(3 0 5) :stop-ids (list first))
        (is (null text))
        (is (eq :stop finish))
        (is (null ids))
        (is (= 3 (local:session-position session)))))))

(test generate-finishes-when-the-context-is-full
  (with-session (session model :n-ctx 5) ()
    (multiple-value-bind (text finish ids) (local:generate session '(3 0 5))
      (declare (ignore text))
      (is (eq :length finish))
      (is (equal (greedy-continuation model '(3 0 5) 2) ids))
      (is (= 5 (local:session-position session))))
    (signals local:context-overflow (local:generate session '(1)))))

(test generate-refuses-what-does-not-fit-or-is-empty
  (with-session (session model :n-ctx 5) ()
    (signals local:context-overflow (local:generate session '(1 2 3 4 5 6)))
    (is (= 0 (local:session-position session)))
    (signals local:session-error (local:generate session '()))
    (signals local:session-error (local:generate session '(1) :stop '("a")))))

(test generate-refuses-an-empty-stop-string
  (let ((tokenizer (fixture-tokenizer)))
    (with-session (session model :tokenizer tokenizer) (:vocab 267)
      (signals local:session-error (local:generate session '(1) :stop '(""))))))

(test generate-ends-on-cancel-or-timeout
  (with-session (session model) ()
    (let ((token (ci:make-cancel-token)) (events nil))
      (ci:cancel token)
      (multiple-value-bind (text finish ids)
          (local:generate session '(3 0 5) :cancel token :sink (lambda (event) (push event events)))
        (declare (ignore text))
        (is (equal '(:error :cancelled) finish))
        (is (null ids))
        (is (equal (list (ci:done nil '(:error :cancelled))) events))))
    (is (equal '(:error :timeout) (nth-value 1 (local:generate session '(3 0 5) :timeout 0))))
    (is (= 0 (local:session-position session)))))

(test generate-checks-the-deadline-between-steps
  (with-session (session model) ()
    (let ((token (ci:make-cancel-token)) (count 0))
      (multiple-value-bind (text finish ids)
          (local:generate session '(3 0 5) :cancel token
                                           :sampler (lambda (logits)
                                                      (when (= 2 (incf count)) (ci:cancel token))
                                                      (local:greedy-sampler logits)))
        (declare (ignore text))
        (is (equal '(:error :cancelled) finish))
        (is (= 2 (length ids)))
        (is (= 5 (local:session-position session)) "every returned token is in the cache")))))

;;; Text

(test generate-streams-text-then-one-done
  (let ((tokenizer (fixture-tokenizer)) (events nil))
    (with-session (session model :tokenizer tokenizer) (:vocab 267)
      (let ((ids (mapcar (lambda (text) (piece tokenizer text)) '("he" "ll" "o"))))
        (multiple-value-bind (text finish out)
            (local:generate session '(1) :sampler (script-sampler ids) :max-tokens 3 :ref :r1
                                         :sink (lambda (event) (push event events)))
          (is (string= "hello" text))
          (is (eq :length finish))
          (is (equal ids out))
          (is (equal (list (ci:text-delta :r1 "he") (ci:text-delta :r1 "ll") (ci:text-delta :r1 "o")
                           (ci:done :r1 :length))
                     (reverse events))))))))

(test generate-stops-at-a-stop-string
  (let ((tokenizer (fixture-tokenizer)))
    (with-session (session model :tokenizer tokenizer) (:vocab 267)
      (let ((ids (mapcar (lambda (text) (piece tokenizer text)) '("he" "ll" "o" "he")))
            (events nil))
        (multiple-value-bind (text finish out)
            (local:generate session '(1) :sampler (script-sampler ids) :stop '("llo")
                                         :sink (lambda (event) (push event events)))
          (is (string= "he" text))
          (is (eq :stop finish))
          (is (= 3 (length out)))
          (is (= 4 (local:session-position session)))
          (is (equal '("he") (loop for event in (reverse events)
                                   when (eq :text-delta (getf event :type)) collect (getf event :text)))))))))

(test generate-releases-a-held-tail-that-never-completes
  (let ((tokenizer (fixture-tokenizer)))
    (with-session (session model :tokenizer tokenizer) (:vocab 267)
      (let ((ids (mapcar (lambda (text) (piece tokenizer text)) '("he" "ll" "o"))))
        (is (string= "hello"
                     (local:generate session '(1) :sampler (script-sampler ids) :max-tokens 3
                                                  :stop '("llx"))))))))

(test generate-releases-a-held-tail-when-the-turn-ends
  (let ((tokenizer (fixture-tokenizer)))
    (with-session (session model :tokenizer tokenizer) (:vocab 267)
      (let ((ids (mapcar (lambda (text) (piece tokenizer text)) '("he" "ll"))))
        (is (string= "hell"
                     (local:generate session '(1) :sampler (script-sampler ids) :max-tokens 2
                                                  :stop '("llo"))))))))

(test generate-renders-messages-through-the-chat-template
  (with-chat-tokenizer (tokenizer :specials '(("bos_token_id" 264) ("eos_token_id" 265)))
    (let ((template (make-instance 'local:chatml)) (messages (list *user*)))
      (with-session (session model :tokenizer tokenizer :chat-template template) (:vocab 274)
        (local:generate session '(1 2) :max-tokens 1)
        (multiple-value-bind (text finish ids)
            (local:generate session messages
                            :sampler (script-sampler (list (local:token-id tokenizer "<|im_end|>"))))
          (is (string= "" text))
          (is (eq :stop finish))
          (is (null ids))
          (is (= (length (local:encode-chat template tokenizer messages))
                 (local:session-position session))
              "messages reset the session"))))
    (with-session (session model :tokenizer tokenizer) (:vocab 274)
      (signals local:session-error (local:generate session (list *user*))))))

;;; Stop strings

(defun split (pending stops)
  (multiple-value-list (local:stop-split pending stops)))

(test stop-split-finds-the-earliest-stop
  (is (equal '("ab" "" t) (split "abXYcd" '("YZ" "XY" "cd"))))
  (is (equal '("a" "" t) (split "aXYb" '("Y" "XY")))))

(test stop-split-holds-back-a-possible-start
  (is (equal '("ab" "XY" nil) (split "abXY" '("XYZ"))))
  (is (equal '("ab" "X" nil) (split "abX" '("XYZ" "qX"))))
  (is (equal '("abXY" "" nil) (split "abXY" '("XZ"))))
  (is (equal '("" "X" nil) (split "X" '("XYZ")))))

(test stop-split-without-stops-releases-everything
  (is (equal '("abc" "" nil) (split "abc" '())))
  (is (equal '("" "" nil) (split "" '("a")))))
