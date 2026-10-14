(in-package #:cl-inference/mlx/tests)
(in-suite :cl-inference/mlx)

(defun mlx-test-object (&rest pairs)
  (let ((table (make-hash-table :test 'equal)))
    (loop for (key value) on pairs by #'cddr do (setf (gethash key table) value))
    table))

(defun write-mlx-reference-fixture (root &key indexed)
  (let* ((directory (merge-pathnames "fixture/" root))
         (metadata (mlx-test-object "architecture" "llama" "precision" "float32"))
         (files (mlx-test-object)))
    (ensure-directories-exist (merge-pathnames "config.json" directory))
    (dolist (name '("config.json" "tokenizer.json" "tokenizer_config.json"))
      (f::write-hf-json directory name (mlx-test-object)))
    (if indexed
        (progn
          (f::write-hf-json directory "model.safetensors.index.json"
                            (mlx-test-object "weight_map"
                                             (mlx-test-object "a" "first.safetensors" "b" "second.safetensors")))
          (dolist (name '("first.safetensors" "second.safetensors"))
            (f::write-octets (merge-pathnames name directory) #(1 2 3))))
        (f::write-octets (merge-pathnames "model.safetensors" directory) #(1 2 3)))
    (let ((oracle (merge-pathnames "fixture-oracle.json" root)))
      (f::write-hf-json root "fixture-oracle.json" (mlx-test-object "metadata" metadata))
      (dolist (path (append (mlx-reference-inputs directory) (list oracle)))
        (setf (gethash (enough-namestring (truename path) (truename root)) files)
              (mlx-reference-sha256 path))))
    (mlx-test-object "fixture" (mlx-test-object "metadata" metadata "files" files))))

(test mlx-reference-verifies-single-and-indexed-inputs
  (dolist (indexed '(nil t))
    (f::with-hf-directory (root)
      (let ((manifest (write-mlx-reference-fixture root :indexed indexed)))
        (is (hash-table-p (verify-mlx-reference "fixture" root manifest)))
        (f::write-hf-json (merge-pathnames "fixture/" root) "chat_template.jinja" "unreviewed template")
        (signals error (verify-mlx-reference "fixture" root manifest))))))

(test mlx-reference-rejects-hash-and-provenance-mismatches
  (f::with-hf-directory (root)
    (let* ((manifest (write-mlx-reference-fixture root))
           (entry (gethash "fixture" manifest)))
      (setf (gethash "metadata" entry) (mlx-test-object "architecture" "qwen3"))
      (signals error (verify-mlx-reference "fixture" root manifest))
      (f::write-hf-json (merge-pathnames "fixture/" root) "config.json" (mlx-test-object "changed" t))
      (signals error (verify-mlx-reference "fixture" root manifest)))))

(test mlx-reference-missing-inputs-and-strict-mode
  (f::with-hf-directory (root)
    (let* ((manifest (write-mlx-reference-fixture root))
           (files (gethash "files" (gethash "fixture" manifest))))
      (setf (gethash "fixture/absent.safetensors" files) (make-string 64 :initial-element #\a))
      (let ((*mlx-acceptance-strict* nil))
        (is (null (verify-mlx-reference "fixture" root manifest))))
      (let ((*mlx-acceptance-strict* t))
        (signals error (verify-mlx-reference "fixture" root manifest)))
      (f::write-hf-json (merge-pathnames "fixture/" root) "config.json" (mlx-test-object "changed" t))
      (let ((*mlx-acceptance-strict* nil))
        (signals error (verify-mlx-reference "fixture" root manifest))))))

(test mlx-reference-rejects-unpinned-shards-and-escaping-paths
  (f::with-hf-directory (root)
    (let* ((manifest (write-mlx-reference-fixture root :indexed t))
           (files (gethash "files" (gethash "fixture" manifest))))
      (remhash "fixture/first.safetensors" files)
      (signals error (verify-mlx-reference "fixture" root manifest))
      (setf (gethash "fixture/../model.safetensors" files) (make-string 64 :initial-element #\a))
      (signals error (verify-mlx-reference "fixture" root manifest)))))

(test mlx-reference-logit-validation
  (multiple-value-bind (maximum relative) (mlx-logit-errors #(1f0 2f0) #(1d0 2d0))
    (is (zerop maximum)) (is (zerop relative)))
  (multiple-value-bind (maximum relative) (mlx-logit-errors #(2f0 2f0) #(1d0 2d0))
    (is (= 1d0 maximum)) (is (< (abs (- relative (sqrt 0.2d0))) 1d-12)))
  (dolist (pair (list (list #(1) #(1 2)) (list #() #()) (list #(0) #(0))
                     (list #("bad") #(1))
                     (list (vector float-features:single-float-positive-infinity) #(1))
                     (list #(1) (vector float-features:single-float-nan))))
    (signals error (mlx-logit-errors (first pair) (second pair)))))

(test mlx-reference-generation-stop-expectation
  (multiple-value-bind (ids finish) (mlx-greedy-expectation #(1 2 3) '(2))
    (is (equal '(1) ids)) (is (eq :stop finish)))
  (multiple-value-bind (ids finish) (mlx-greedy-expectation #(2 1 3) '(2))
    (is (null ids)) (is (eq :stop finish)))
  (multiple-value-bind (ids finish) (mlx-greedy-expectation #(1 2 3) '(4))
    (is (equal '(1 2 3) ids)) (is (eq :length finish))))
