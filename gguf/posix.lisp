(in-package #:cl-inference/gguf)

(defconstant +header-bytes+ 24
  "Magic, version, tensor count and metadata count.")

(defun map-file (path)
  (cl-inference/mapping:map-file path +header-bytes+ #'gguf-fail))

(defun unmap-file (pointer size)
  (cl-inference/mapping:unmap-file pointer size #'gguf-fail))
