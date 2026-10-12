(in-package #:cl-inference/local/tests)

(in-suite :cl-inference/local)

;;; llama.cpp's models/ggml-vocab-*.gguf with their .inp and .out files, kept in the
;;; gitignored reference/ directory.

(defun fixture-path (name)
  (asdf:system-relative-pathname :cl-inference (format nil "reference/~A" name)))

(defun fixture-text (path)
  (babel:octets-to-string (read-octets path) :encoding :utf-8))

(defun split-on (separator string)
  (loop with start = 0
        for hit = (search separator string :start2 start)
        collect (subseq string start hit)
        while hit
        do (setf start (+ hit (length separator)))))

(defun parse-id-line (line)
  (let ((ids '()) (start 0))
    (loop while (< start (length line))
          do (multiple-value-bind (id end) (parse-integer line :start start :junk-allowed t)
               (when id (push id ids))
               (setf start (if end (1+ end) (length line)))))
    (nreverse ids)))

(defun check-vocab-fixture (name expected-class)
  (let ((gguf (fixture-path (format nil "ggml-vocab-~A.gguf" name))))
    (if (not (probe-file gguf))
        (skip "~A is not in reference/" name)
        (let ((inputs (split-on (format nil "~%__ggml_vocab_test__~%")
                                (fixture-text (format nil "~A.inp" (namestring gguf)))))
              (expected (mapcar #'parse-id-line
                                (split-on (string #\Newline)
                                          (fixture-text (format nil "~A.out" (namestring gguf)))))))
          (local:with-weights (w gguf)
            (let ((tokenizer (local:load-tokenizer w)))
              (is (typep tokenizer expected-class))
              (loop for input in inputs for want in expected
                    do (is (equal want (coerce (local:encode tokenizer input :add-bos nil) 'list))
                           "~S encodes differently from llama.cpp" input))))))))

(test llama-bpe-matches-llama-cpp-vocab-fixture
  (check-vocab-fixture "llama-bpe" 'local:llama-bpe))

(test qwen2-matches-llama-cpp-vocab-fixture
  (check-vocab-fixture "qwen2" 'local:qwen2))
