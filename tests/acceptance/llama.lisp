(in-package #:cl-inference/acceptance)

(in-suite :cl-inference/acceptance)

;;; Llama 3.2 1B Instruct against llama.cpp. The model and the oracle's dump live in the
;;; gitignored reference/ directory; every test skips without them.

(defparameter *cases*
  '((:f16 :model :f16 :text "The capital of France is")
    (:f16-chat :model :f16 :date "26 Jul 2024"
     :messages ((:role :system :content "Answer in one word.")
                (:role :user :content "What is the capital of France?"))))
  "Each case reads its dump from reference/oracle/<name>/, made by reference/llama-oracle.sh.")

(defun case-option (case key) (getf (rest (assoc case *cases*)) key))

(defparameter *greedy-horizon* 4
  "Short, because one forward takes about a minute until half-precision matmul lands (#4).")

;;; llama.cpp multiplies in f16 or q8 activations; ours stay f32. Observed worst case: cosine
;;; 0.999992, relative error 0.0040 on 6 tokens and 0.0119 on 46.
(defparameter *minimum-cosine* 0.9999d0)
(defparameter *maximum-relative-error* 0.02d0)

;; TODO: Q8_0 acceptance (bartowski's Llama-3.2-1B-Instruct-Q8_0.gguf) once the dtype lands (#3)
(defparameter *models*
  '((:f16 "Llama-3.2-1B-Instruct-f16.gguf"
     "1f33ad43d2b85b908ff06fe7002b69806a57359b9b2617ca27d7bdea428ae146")))

(defun reference-path (&rest parts)
  (asdf:system-relative-pathname "cl-inference" (format nil "reference/~{~a~^/~}~:[/~;~]" parts (find #\. (car (last parts))))))

(defun model-path (kind) (reference-path "models" (second (assoc kind *models*))))
(defun model-sha (kind) (third (assoc kind *models*)))

(defun sha256 (path)
  (let ((output (handler-case (uiop:run-program (list "shasum" "-a" "256" (namestring path))
                                                :output :string)
                  (error () (uiop:run-program (list "sha256sum" (namestring path))
                                              :output :string)))))
    (subseq output 0 64)))

(defvar *hashes* (make-hash-table))

(defun verified-model-p (kind)
  "Whether the model for KIND is present and has its pinned sha256."
  (let ((path (model-path kind)))
    (and (probe-file path)
         (multiple-value-bind (hash found) (gethash kind *hashes*)
           (string= (model-sha kind)
                    (if found hash (setf (gethash kind *hashes*) (sha256 path))))))))

;;; The oracle's dump: prompt ids, greedy ids, top logits and the last position of each layer

(defun read-lines (path) (uiop:read-file-lines path))

(defun parse-floats (strings)
  (let ((*read-default-float-format* 'single-float))
    (map 'vector (lambda (string) (coerce (read-from-string string) 'double-float)) strings)))

(defun oracle (case)
  "The oracle's dump for CASE as a plist, or NIL when it has not been generated."
  (let ((directory (reference-path "oracle" (string-downcase (symbol-name case)))))
    (when (probe-file (merge-pathnames "trace.txt" directory))
      (flet ((ids (name)
               (mapcar #'parse-integer (read-lines (merge-pathnames name directory)))))
        (list :prompt (ids "prompt.ids") :greedy (ids "greedy.ids")
              :top (loop for line in (read-lines (merge-pathnames "top.txt" directory))
                         for (id logit) = (uiop:split-string line :separator " ")
                         collect (cons (parse-integer id) (aref (parse-floats (list logit)) 0)))
              :trace (loop for line in (read-lines (merge-pathnames "trace.txt" directory))
                           for (name count . values) = (uiop:split-string line :separator " ")
                           collect (cons name (parse-floats values))))))))

;;; Our run: one traced prefill and one generate, shared by the tests

(defvar *runs* (make-hash-table))

(defun vector-of-row (tensor row)
  (let ((columns (aref (ct:tensor-shape tensor) 1)))
    (coerce (loop for j below columns collect (coerce (ct:tref tensor row j) 'double-float))
            'vector)))

(defun compute-run (case)
  (local:with-weights (weights (model-path (case-option case :model)))
    (let* ((model (local:load-model weights))
           (tokenizer (local:load-tokenizer weights))
           (template (local:load-chat-template weights))
           (messages (case-option case :messages))
           (prompt (coerce (if messages
                               (local:encode-chat template tokenizer messages
                                                  :date (case-option case :date))
                               (local:encode tokenizer (case-option case :text)))
                           'list))
           (session (local:make-session model :tokenizer tokenizer :chat-template template))
           (start (get-internal-real-time))
           (logits nil)
           (trace (tests::tracing-layers (setf logits (local:session-prefill session prompt)))))
      (local:session-reset session)
      (multiple-value-bind (text finish ids)
          (local:generate session prompt :max-tokens *greedy-horizon* :timeout 3600000)
        (declare (ignore text))
        (list :prompt prompt
              :prefill-seconds (/ (- (get-internal-real-time) start) internal-time-units-per-second)
              :layers (mapcar (lambda (tensor)
                                (vector-of-row tensor (1- (aref (ct:tensor-shape tensor) 0))))
                              trace)
              :logits (vector-of-row logits 0)
              :stop-ids (local:chat-template-stop-ids template tokenizer)
              :finish finish
              :greedy ids)))))

(defun our-run (case)
  (or (gethash case *runs*) (setf (gethash case *runs*) (compute-run case))))

;;; Comparison

(defun dot (a b) (reduce #'+ (map 'vector #'* a b)))
(defun cosine (a b) (/ (dot a b) (sqrt (* (dot a a) (dot b b)))))
(defun relative-error (a b)
  (let ((diff (map 'vector #'- a b))) (sqrt (/ (dot diff diff) (dot b b)))))

(defun top-ids (logits count)
  (map 'list #'car (subseq (sort (coerce (loop for logit across logits for id from 0
                                               collect (cons id logit))
                                         'vector)
                                 #'> :key #'cdr)
                           0 count)))

(defmacro with-acceptance ((case (ours oracle)) &body body)
  `(let ((model (case-option ,case :model)))
     (cond ((not (verified-model-p model))
            (skip "~A is not in reference/models with its pinned sha256"
                  (second (assoc model *models*))))
           ((not (oracle ,case))
            (skip "run reference/llama-oracle.sh for ~A first" ,case))
           (t (let ((,ours (our-run ,case)) (,oracle (oracle ,case)))
                ,@body)))))

;;; Metadata

(test llama-3-2-1b-f16-metadata
  (if (not (verified-model-p :f16))
      (skip "Llama-3.2-1B-Instruct-f16.gguf is not in reference/models")
      (local:with-weights (weights (model-path :f16))
        (let ((architecture (local:load-architecture weights)))
          (is (eq 'local:llama (type-of architecture)))
          (is (= 500000 (round (local:hparam architecture 'rope-theta))))
          (is (= 16 (local:hparam architecture 'n-layers)))
          (is (member "rope_freqs.weight" (local:weights-tensor-names weights) :test #'string=))
          (is (not (member "output.weight" (local:weights-tensor-names weights) :test #'string=)))
          (is (equal "llama-bpe" (local:weights-metadata weights "tokenizer.ggml.pre")))
          (is (eq 'local:llama-bpe (type-of (local:load-tokenizer weights))))
          (is (eq 'local:llama3-chat (type-of (local:load-chat-template weights))))
          (is (member 128009 (local:chat-template-stop-ids (local:load-chat-template weights)
                                                           (local:load-tokenizer weights))))))))

;;; Against llama.cpp

(defun check-close (name ours expected)
  (is (>= (cosine ours expected) *minimum-cosine*) "~A cosine ~,6f" name (cosine ours expected))
  (is (<= (relative-error ours expected) *maximum-relative-error*)
      "~A relative error ~,5f" name (relative-error ours expected)))

(defun without-stop-ids (ids stop-ids)
  (remove-if (lambda (id) (member id stop-ids)) ids))

(defmacro define-case-tests (case)
  (flet ((name (suffix) (intern (format nil "~A-~A" case suffix))))
    `(progn
       (test ,(name "PROMPT-IDS")
         (with-acceptance (,case (ours oracle))
           (is (equal (getf oracle :prompt) (getf ours :prompt)))))
       (test ,(name "LAYER-TRACES")
         (with-acceptance (,case (ours oracle))
           (is (= 16 (length (getf ours :layers))))
           (loop for layer in (getf ours :layers) for index from 0
                 for name = (format nil "l_out-~D" index)
                 do (check-close name layer
                                 (cdr (assoc name (getf oracle :trace) :test #'string=))))))
       (test ,(name "LOGITS")
         (with-acceptance (,case (ours oracle))
           (check-close "logits" (getf ours :logits)
                        (cdr (assoc "result_output" (getf oracle :trace) :test #'string=)))
           (is (equal (subseq (mapcar #'car (getf oracle :top)) 0 5)
                      (top-ids (getf ours :logits) 5)))))
       (test ,(name "GREEDY")
         (with-acceptance (,case (ours oracle))
           (let ((expected (subseq (getf oracle :greedy) 0
                                   (min *greedy-horizon* (length (getf oracle :greedy)))))
                 (stops (getf ours :stop-ids)))
             (is (equal (without-stop-ids expected stops)
                        (without-stop-ids (getf ours :greedy) stops)))
             ;; the reply ends where llama.cpp's does: on a stop id, or at the horizon
             (is (eq (if (intersection expected stops) :stop :length) (getf ours :finish)))))))))

(define-case-tests :f16)
(define-case-tests :f16-chat)
