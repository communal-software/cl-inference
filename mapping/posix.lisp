;;; File mapping adapted from cl-qwen src/gguf.lisp (MIT, see gguf/reader.lisp).
(in-package #:cl-inference/mapping)

#-(or darwin linux freebsd openbsd netbsd dragonfly)
(error "cl-inference file mapping requires POSIX mmap")

(defconstant +o-rdonly+ 0)
(defconstant +seek-end+ 2)
(defconstant +prot-read+ 1)
(defconstant +prot-write+ 2)
(defconstant +map-private+ 2)

(cffi:defcfun ("open" posix-open) :int (path :string) (flags :int))
(cffi:defcfun ("close" posix-close) :int (fd :int))
(cffi:defcfun ("lseek" posix-lseek) :int64 (fd :int) (offset :int64) (whence :int))
(cffi:defcfun ("mmap" posix-mmap) :pointer
  (address :pointer) (length :size) (protection :int) (flags :int) (fd :int) (offset :int64))
(cffi:defcfun ("munmap" posix-munmap) :int (address :pointer) (length :size))

(defun map-file (path minimum-size fail)
  "Map PATH copy-on-write; FAIL signals the caller's format-specific condition."
  (let ((fd (posix-open (namestring path) +o-rdonly+)))
    (when (minusp fd) (funcall fail "Cannot open ~A" path))
    (unwind-protect
         (let ((size (posix-lseek fd 0 +seek-end+)))
           (when (< size (max 1 minimum-size))
             (funcall fail "~A is too short (minimum ~D bytes)" path minimum-size))
           (let ((pointer (posix-mmap (cffi:null-pointer) size
                                      (logior +prot-read+ +prot-write+) +map-private+ fd 0)))
             (when (= (cffi:pointer-address pointer)
                      (1- (ash 1 (* 8 (cffi:foreign-type-size :pointer)))))
               (funcall fail "Cannot map ~A" path))
             (values pointer size)))
      (posix-close fd))))

(defun unmap-file (pointer size fail)
  (unless (zerop (posix-munmap pointer size))
    (funcall fail "Cannot unmap file")))
