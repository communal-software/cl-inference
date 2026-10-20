(in-package #:cl-inference/acceptance)

(in-suite :cl-inference/acceptance)

(defvar *cases* nil)
(defvar *models* nil)
(defvar *reference-directory* nil)

(defun case-option (case key) (getf (rest (assoc case *cases*)) key))

(defparameter *greedy-horizon* 8
  "The ids the oracle dumps.")

;;; Minimum cosine, maximum relative error and ordered top-logit count, per weight dtype.
(defparameter *tolerances* '((:f16 0.9999d0 0.02d0 5) (:q8_0 0.999d0 0.04d0 3)))

(defun case-tolerances (case)
  (or (case-option case :tolerances)
      (rest (assoc (model-option (case-option case :model) :dtype) *tolerances*))))

(defun reference-path (&rest parts)
  (merge-pathnames
   (format nil "~{~a~^/~}~:[/~;~]" parts (find #\. (car (last parts))))
   (or *reference-directory* (asdf:system-relative-pathname "cl-inference" "reference/"))))

(defun model-path (kind) (reference-path "models" (second (assoc kind *models*))))
(defun model-sha (kind) (third (assoc kind *models*)))
(defun model-option (kind key) (getf (cdddr (assoc kind *models*)) key))
(defun case-horizon (case) (or (case-option case :horizon) *greedy-horizon*))

(defun acceptance-test-names (cases)
  (unless cases (error "Acceptance case selection is empty"))
  (let ((names (remove-duplicates
                (loop for case in cases
                      unless (assoc case *cases*) do (error "Unknown acceptance case: ~S" case)
                      append (append
                              (let ((metadata (model-option (case-option case :model) :metadata-test)))
                                (when metadata (list metadata)))
                              (loop for suffix in '(prompt-ids layer-traces logits greedy)
                                    collect (intern (format nil "~A-~A" case suffix)
                                                    :cl-inference/acceptance))))
                :from-end t)))
    (dolist (name names)
      (unless (get-test name) (error "Missing acceptance test: ~S" name)))
    names))

(defun run-tests (&key (cases nil cases-p) strict)
  (clrhash *hashes*)
  (clrhash *runs*)
  (let ((selection (if cases-p (acceptance-test-names cases) :cl-inference/acceptance)))
    (when strict
      (dolist (case (if cases-p cases (mapcar #'first *cases*)))
        (unless (verified-model-p (case-option case :model))
          (error "~A: model missing or SHA256 differs" case))
        (unless (oracle case) (error "~A: oracle dump missing" case))
        (verify-case-oracle (case-option case :family) case)))
    (let ((results (run selection)))
      (explain! results)
      (acceptance-results-ok-p results strict))))

(defun acceptance-results-ok-p (results strict)
  (multiple-value-bind (passed failures skipped) (results-status results)
    (declare (ignore failures))
    (and passed (or (not strict) (and results (null skipped))))))

(defun sha256 (path)
  (let ((output (handler-case (uiop:run-program (list "shasum" "-a" "256" (namestring path))
                                                :output :string)
                  (error () (uiop:run-program (list "sha256sum" (namestring path))
                                              :output :string)))))
    (subseq output 0 64)))

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
  (let ((*read-default-float-format* 'single-float) (*read-eval* nil))
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
                           do (unless (= (parse-integer count) (length values))
                                (error "Malformed oracle trace ~A" name))
                           collect (cons name (parse-floats values))))))))

;;; Our run: one traced prefill and one generate, shared by the tests

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
                                                  :date (case-option case :date)
                                                  :enable-thinking (case-option case :thinking))
                               (local:encode tokenizer (case-option case :text)))
                           'list))
           (session (local:make-session model :tokenizer tokenizer :chat-template template
                                       :n-batch (or (case-option case :n-batch) 512)))
           (start (get-internal-real-time))
           (logits nil)
           (trace (tests::tracing-layers (setf logits (local:session-prefill session prompt)))))
      (local:session-reset session)
      (multiple-value-bind (text finish ids)
          (local:generate session prompt :max-tokens (case-horizon case) :timeout 3600000)
        (declare (ignore text))
        (list :prompt prompt
              :seconds (/ (- (get-internal-real-time) start) internal-time-units-per-second)
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

(defun dot (a b)
  (unless (= (length a) (length b))
    (error "Trace dimensions differ: ~D and ~D" (length a) (length b)))
  (reduce #'+ (map 'vector #'* a b)))
(defun cosine (a b) (/ (dot a b) (sqrt (* (dot a a) (dot b b)))))
(defun relative-error (a b)
  (let ((diff (map 'vector #'- a b))) (sqrt (/ (dot diff diff) (dot b b)))))

(defun top-ids (logits count)
  (map 'list #'car (subseq (sort (coerce (loop for logit across logits for id from 0
                                               collect (cons id logit))
                                         'vector)
                                 #'> :key #'cdr)
                           0 count)))

(defun unavailable-acceptance (case detail)
  (if (or (and (case-option case :quant) (equal "1" (uiop:getenv "CL_INFERENCE_QUANT_ACCEPTANCE_STRICT")))
          (let ((variable (case-option case :strict-env)))
            (and variable (equal "1" (uiop:getenv variable)))))
      (error "~A: ~A" case detail)
      (skip "~A: ~A" case detail)))

(defun verify-quant-oracle (case)
  (when (case-option case :quant)
    (let* ((key (string-downcase (symbol-name (case-option case :model))))
           (expected (gethash key *quant-manifest*))
           (path (reference-path "oracle" (string-downcase (symbol-name case)) "provenance.json")))
      (unless (probe-file path) (error "Missing oracle provenance for ~A" case))
      (let ((actual (com.inuoe.jzon:parse path)))
        (dolist (field '("sha256" "source_sha256" "llama_cpp" "dtype"))
          (unless (equal (gethash field expected) (gethash field actual))
            (error "Oracle provenance mismatch for ~A: ~A" case field)))
        (multiple-value-bind (native present) (gethash "native" actual)
          (unless (and present (null native))
            (error "~A requires the F32-dequantized oracle" case)))))))

(defgeneric verify-case-oracle (family case)
  (:method (family case)
    (declare (ignore family))
    (verify-quant-oracle case)))

(defmacro with-acceptance ((case (ours oracle)) &body body)
  `(let ((model (case-option ,case :model)))
     (cond ((and (or (case-option ,case :quant) (case-option ,case :strict-env))
                 (probe-file (model-path model)) (not (verified-model-p model)))
            (error "~A model SHA256 differs from its pin" ,case))
           ((not (verified-model-p model))
            (unavailable-acceptance ,case "Model missing or SHA256 differs"))
           ((not (oracle ,case))
            (unavailable-acceptance ,case "Oracle dump missing"))
           (t (verify-case-oracle (case-option ,case :family) ,case)
              (let ((,ours (our-run ,case)) (,oracle (oracle ,case)))
                ,@body)))))

;;; Against llama.cpp

(defun check-close (name ours expected tolerances)
  (destructuring-bind (minimum-cosine maximum-error &rest counts)
      tolerances
    (declare (ignore counts))
    (is (>= (cosine ours expected) minimum-cosine) "~A cosine ~,6f" name (cosine ours expected))
    (is (<= (relative-error ours expected) maximum-error)
        "~A relative error ~,5f" name (relative-error ours expected))))

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
           (is (= (case-option ,case :layers) (length (getf ours :layers))))
           (loop for layer in (getf ours :layers) for index from 0
                 for name = (format nil "l_out-~D" index)
                 do (check-close name layer
                                 (cdr (assoc name (getf oracle :trace) :test #'string=))
                                 (case-tolerances ,case)))))
       (test ,(name "LOGITS")
         (with-acceptance (,case (ours oracle))
           (check-close "logits" (getf ours :logits)
                        (cdr (assoc "result_output" (getf oracle :trace) :test #'string=))
                        (case-tolerances ,case))
           (let ((count (third (case-tolerances ,case))))
             (is (equal (subseq (mapcar #'car (getf oracle :top)) 0 count)
                        (top-ids (getf ours :logits) count)))
             (let ((unordered (case-option ,case :unordered-top)))
               (when unordered
                 (is (equal (sort (subseq (mapcar #'car (getf oracle :top)) 0 unordered) #'<)
                            (sort (top-ids (getf ours :logits) unordered) #'<))))))))
       (test ,(name "GREEDY")
         (with-acceptance (,case (ours oracle))
           (let ((expected (subseq (getf oracle :greedy) 0
                                   (min (case-horizon ,case) (length (getf oracle :greedy)))))
                 (stops (getf ours :stop-ids)))
             (is (equal (without-stop-ids expected stops)
                        (without-stop-ids (getf ours :greedy) stops)))
             (is (eq (if (intersection expected stops) :stop :length) (getf ours :finish)))))))))
