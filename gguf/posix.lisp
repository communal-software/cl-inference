;;; File mapping adapted from cl-qwen src/gguf.lisp (MIT, see reader.lisp).
(in-package #:cl-inference/gguf)

#-(or darwin linux freebsd openbsd netbsd dragonfly)
(error "cl-inference/gguf maps files with POSIX mmap; this platform is not supported")

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

(defconstant +header-bytes+ 24
  "Magic, version, tensor count and metadata count.")

(defun map-failed-p (pointer)
  (= (cffi:pointer-address pointer) (1- (ash 1 (* 8 (cffi:foreign-type-size :pointer))))))

(defun map-file (path)
  "Map PATH copy-on-write. Return the base pointer and the file size."
  (let ((fd (posix-open (namestring path) +o-rdonly+)))
    (when (minusp fd)
      (gguf-fail "Cannot open ~A" path))
    (unwind-protect
         (let ((size (posix-lseek fd 0 +seek-end+)))
           (when (< size +header-bytes+)
             (gguf-fail "~A is too short to be a GGUF file" path))
           (let ((pointer (posix-mmap (cffi:null-pointer) size
                                      (logior +prot-read+ +prot-write+) +map-private+ fd 0)))
             (when (map-failed-p pointer)
               (gguf-fail "Cannot map ~A" path))
             (values pointer size)))
      (posix-close fd))))

(defun unmap-file (pointer size)
  (unless (zerop (posix-munmap pointer size))
    (gguf-fail "Cannot unmap GGUF file")))
