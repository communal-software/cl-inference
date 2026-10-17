(in-package #:cl-inference/acceptance)

(def-suite :cl-inference/gemma/acceptance :in :cl-inference/acceptance)
(in-suite :cl-inference/gemma/acceptance)

(defparameter *gemma-manifest*
  (com.inuoe.jzon:parse (asdf:system-relative-pathname "cl-inference" "tests/acceptance/gemma-model.json")))

(pushnew (list :gemma-q8 (gethash "file" *gemma-manifest*) (gethash "sha256" *gemma-manifest*) :dtype :q8_0)
         *models* :key #'first)

(maphash
 (lambda (name config)
   (let ((key (intern (string-upcase (concatenate 'string "gemma-" name)) :keyword)))
     (pushnew
      (append (list key :model :gemma-q8 :layers 18 :family :gemma :n-batch 1024
                    :strict-env "CL_INFERENCE_GEMMA_ACCEPTANCE_STRICT" :tolerances '(0.9999d0 0.02d0 5))
              (if (gethash "messages" config)
                  (list :messages (loop for message across (gethash "messages" config)
                                        collect (list :role (intern (string-upcase (gethash "role" message)) :keyword)
                                                      :content (gethash "content" message))))
                  (list :text (or (gethash "text" config)
                                  (concatenate 'string
                                    (format nil "~{~A~}" (make-list (gethash "count" config)
                                                                    :initial-element (gethash "repeat" config)))
                                    (gethash "suffix" config))))))
      *cases* :key #'first)))
 (gethash "cases" *gemma-manifest*))

(defmethod verify-case-oracle ((family (eql :gemma)) case)
  (let* ((name (string-downcase (symbol-name case)))
         (directory (reference-path "oracle" name))
         (path (merge-pathnames "provenance.json" directory)))
    (dolist (file '("provenance.json" "decode.f32" "prompt.ids" "greedy.ids" "top.txt" "trace.txt"))
      (unless (probe-file (merge-pathnames file directory)) (error "~A: missing ~A" case file)))
    (let ((actual (com.inuoe.jzon:parse path)))
      (dolist (field '("sha256" "revision" "llama_cpp" "dtype" "reference_sha256"))
        (unless (equal (gethash field *gemma-manifest*) (gethash field actual))
          (error "~A: oracle provenance mismatch for ~A" case field)))
      (unless (equalp (gethash (subseq name 6) (gethash "cases" *gemma-manifest*)) (gethash "case" actual))
        (error "~A: oracle prompt provenance mismatch" case))
      (multiple-value-bind (native present) (gethash "native" actual)
        (unless (and present (null native)) (error "~A requires an F32-dequantized oracle" case))))))

(define-case-tests :gemma-plain)
(define-case-tests :gemma-chat)
(define-case-tests :gemma-system)
(define-case-tests :gemma-multi)
(define-case-tests :gemma-window)

(test gemma-model-metadata-and-client
  (with-acceptance (:gemma-chat (ours oracle))
    (declare (ignore ours oracle))
    (local:with-weights (weights (model-path :gemma-q8))
      (let ((architecture (local:load-architecture weights))
            (backend (local:make-local-backend weights :n-ctx 128)))
        (unwind-protect
             (progn
               (is (typep architecture 'local:gemma3))
               (dolist (entry '((n-layers 18) (dim 640) (n-heads 4) (n-kv-heads 1)
                                (head-dim 256) (sliding-window 512) (context-length 32768)))
                 (is (= (second entry) (local:hparam architecture (first entry)))))
               (is (member :q8_0 (mapcar (lambda (name)
                                          (local:tensor-info-format-type (local:weights-tensor-info weights name)))
                                        (local:weights-tensor-names weights))))
               (let ((result (ci:complete backend :messages (case-option :gemma-chat :messages)
                                                 :temperature 0 :max-tokens 8 :timeout 3600000)))
                 (is (not (ci:result-error-p result)))
                 (is (stringp (ci:content-text (getf (second result) :content))))))
          (local:close-local-backend backend))))))

(test gemma-window-crosses-real-attention-boundary
  (with-acceptance (:gemma-window (ours oracle))
    (declare (ignore oracle))
    (is (> (length (getf ours :prompt)) 512))))

(test gemma-chunked-prefill-and-cached-decode
  (dolist (case '(:gemma-chat :gemma-window))
    (with-acceptance (case (ours oracle))
      (local:with-weights (weights (model-path :gemma-q8))
        (let* ((model (local:load-model weights))
               (session (local:make-session model :n-ctx 1024 :n-batch 128))
               (vocab (local:tokenizer-vocab-size (local:load-tokenizer weights)))
               (ids (without-stop-ids (getf oracle :greedy) (getf ours :stop-ids)))
               (path (reference-path "oracle" (string-downcase (symbol-name case)) "decode.f32")))
          (unwind-protect
               (with-open-file (stream path :element-type '(unsigned-byte 8))
                 (is (= (* 4 vocab (1+ (length ids))) (file-length stream)))
                 (loop with logits = (local:session-prefill session (getf ours :prompt))
                       for remaining on (cons nil ids)
                       for expected = (make-array vocab :element-type 'single-float)
                       for bytes = (make-array (* 4 vocab) :element-type '(unsigned-byte 8))
                       do (read-sequence bytes stream)
                          (cffi:with-pointer-to-vector-data (pointer bytes)
                            (dotimes (i vocab) (setf (aref expected i) (cffi:mem-aref pointer :float i))))
                          (check-close "cached logits" (vector-of-row logits 0)
                                       (map 'vector (lambda (x) (coerce x 'double-float)) expected)
                                       '(0.9999d0 0.02d0 5))
                          (when (rest remaining)
                            (setf logits (local:session-prefill session (list (second remaining)))))))
            (local:close-session session)))))))
