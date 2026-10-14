(in-package #:cl-inference/examples)

(defun inspect-weights (path)
  "Return format, architecture and tensor summaries without loading a model."
  (local:with-weights (weights path)
    (list :format (local:weights-format weights)
          :architecture (local:weights-architecture-name weights)
          :tensors (loop for name in (local:weights-tensor-names weights)
                         for info = (local:weights-tensor-info weights name)
                         collect (list :name name :shape (coerce (local:tensor-info-shape info) 'list)
                                       :type (local:tensor-info-format-type info)
                                       :bytes (local:tensor-info-byte-count info))))))
