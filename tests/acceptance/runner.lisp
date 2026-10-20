(in-package #:cl-inference/acceptance)

(def-suite :cl-inference/acceptance-runner-fixtures)
(in-suite :cl-inference/acceptance-runner-fixtures)

(defvar *runner-skip* nil)
(defvar *runner-fail* nil)

(test runner-metadata (is (not *runner-fail*)))
(test runner-prompt-ids (is (not *runner-fail*)))
(test runner-layer-traces (is (not *runner-fail*)))
(test runner-logits
  (if *runner-skip* (skip "Fixture skip") (is (not *runner-fail*))))
(test runner-greedy (is (not *runner-fail*)))

(def-suite :cl-inference/acceptance-runner)
(in-suite :cl-inference/acceptance-runner)

(defmacro with-runner-inputs (&body body)
  `(tests::with-hf-directory (directory)
     (let* ((*reference-directory* directory)
            (*cases* '((:runner :model :runner-model)))
            (model (reference-path "models" "fixture.gguf"))
            (oracle (reference-path "oracle" "runner"))
            (*hashes* (make-hash-table))
            (*runs* (make-hash-table)))
       (ensure-directories-exist model)
       (with-open-file (stream model :direction :output) (write-string "fixture" stream))
       (ensure-directories-exist (merge-pathnames "trace.txt" oracle))
       (loop for (name text) in '(("trace.txt" "result_output 1 1.0")
                                 ("prompt.ids" "1") ("greedy.ids" "2") ("top.txt" "2 1.0"))
             do (with-open-file (stream (merge-pathnames name oracle) :direction :output)
                  (write-line text stream)))
       (let ((*models* (list (list :runner-model "fixture.gguf" (sha256 model)
                                   :metadata-test 'runner-metadata)))
             (*test-dribble* (make-broadcast-stream)))
         ,@body))))

(test selection-rejects-unknown-empty-and-unregistered-cases
  (signals error (run-tests :cases nil))
  (signals error (run-tests :cases '(:not-a-case)))
  (let ((*cases* '((:unregistered :model :runner-model))))
    (signals error (run-tests :cases '(:unregistered)))))

(test selection-includes-only-requested-cases-and-shared-metadata-once
  (let* ((*cases* '((:qwen-f16 :model :qwen-f16)
                   (:qwen-f16-chat :model :qwen-f16)))
         (names (acceptance-test-names '(:qwen-f16 :qwen-f16-chat))))
    (is (= 9 (length names)))
    (is (= 1 (count 'qwen-f16-metadata names)))
    (is (member 'qwen-f16-chat-greedy names))
    (is (not (member 'qwen-q8-greedy names)))))

(test strict-runner-runs-selected-tests-and-clears-caches
  (with-runner-inputs
    (setf (gethash :stale *hashes*) t (gethash :stale *runs*) t)
    (is (run-tests :cases '(:runner) :strict t))
    (is (not (gethash :stale *hashes*)))
    (is (not (gethash :stale *runs*)))))

(test strict-runner-rejects-missing-and-mismatched-models
  (with-runner-inputs
    (setf (second (first *models*)) "missing.gguf")
    (signals error (run-tests :cases '(:runner) :strict t))
    (setf (second (first *models*)) "fixture.gguf"
          (third (first *models*)) (make-string 64 :initial-element #\0))
    (signals error (run-tests :cases '(:runner) :strict t))))

(test strict-runner-rejects-missing-oracles
  (with-runner-inputs
    (setf *reference-directory* (merge-pathnames "absent/" directory))
    (ensure-directories-exist (reference-path "models" "fixture.gguf"))
    (uiop:copy-file model (reference-path "models" "fixture.gguf"))
    (signals error (run-tests :cases '(:runner) :strict t))))

(test strict-runner-fails-skips-and-failures
  (with-runner-inputs
    (let ((*runner-skip* t))
      (is (run-tests :cases '(:runner)))
      (is (not (run-tests :cases '(:runner) :strict t))))
    (let ((*runner-fail* t))
      (is (not (run-tests :cases '(:runner))))
      (is (not (run-tests :cases '(:runner) :strict t)))))
  (is (not (acceptance-results-ok-p nil t))))

(test ordinary-missing-acceptance-still-skips
  (tests::with-hf-directory (directory)
    (let ((*reference-directory* directory)
          (*test-dribble* (make-broadcast-stream)))
      (is (run-tests :cases '(:qwen-f16))))))
