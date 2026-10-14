;;; Adapted from cl-qwen src/tokenizer.lisp
;;;
;;; MIT License
;;;
;;; Copyright (c) 2026 George Watson
;;;
;;; Permission is hereby granted, free of charge, to any person obtaining a copy
;;; of this software and associated documentation files (the "Software"), to deal
;;; in the Software without restriction, including without limitation the rights
;;; to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
;;; copies of the Software, and to permit persons to whom the Software is
;;; furnished to do so, subject to the following conditions:
;;;
;;; The above copyright notice and this permission notice shall be included in all
;;; copies or substantial portions of the Software.
;;;
;;; THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
;;; IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
;;; FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
;;; AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
;;; LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
;;; OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
;;; SOFTWARE.

(in-package #:cl-inference/local)

;;; Merging works on token ids: a candidate is a merge-table lookup of two neighbouring ids, so
;;; no strings are built. Heap keys are rank*2^32 + left position, which orders by rank and then
;;; leftmost. A popped key is stale when its pair no longer has that rank, and is dropped.

(defstruct (bpe-work (:constructor make-bpe-work ()))
  (bytes (make-array 256 :element-type '(unsigned-byte 8)) :type (simple-array (unsigned-byte 8) (*)))
  (ids (make-array 256 :element-type 'fixnum) :type (simple-array fixnum (*)))
  (next (make-array 256 :element-type 'fixnum) :type (simple-array fixnum (*)))
  (previous (make-array 256 :element-type 'fixnum) :type (simple-array fixnum (*)))
  (heap (make-array 64 :element-type 'fixnum :adjustable t :fill-pointer 0)
   :type (and (vector fixnum) (not simple-array))))

(defun bpe-work-reserve (work size)
  (when (< (length (bpe-work-ids work)) size)
    (let ((size (max size (* 2 (length (bpe-work-ids work))))))
      (setf (bpe-work-bytes work) (make-array size :element-type '(unsigned-byte 8))
            (bpe-work-ids work) (make-array size :element-type 'fixnum)
            (bpe-work-next work) (make-array size :element-type 'fixnum)
            (bpe-work-previous work) (make-array size :element-type 'fixnum)))))

(defun heap-push (heap key)
  (declare (type (and (vector fixnum) (not simple-array)) heap) (fixnum key))
  (vector-push-extend key heap)
  (loop for child of-type fixnum = (1- (length heap)) then parent
        for parent of-type fixnum = (ash (1- child) -1)
        while (and (> child 0) (< (aref heap child) (aref heap parent)))
        do (rotatef (aref heap child) (aref heap parent))))

(defun heap-pop (heap)
  (declare (type (and (vector fixnum) (not simple-array)) heap))
  (let ((top (aref heap 0)) (last (vector-pop heap)))
    (declare (fixnum top last))
    (when (plusp (length heap))
      (setf (aref heap 0) last)
      (loop with n of-type fixnum = (length heap)
            for parent of-type fixnum = 0 then child
            for left of-type fixnum = (1+ (* 2 parent))
            while (< left n)
            for child of-type fixnum = (if (and (< (1+ left) n) (< (aref heap (1+ left)) (aref heap left)))
                                           (1+ left)
                                           left)
            while (< (aref heap child) (aref heap parent))
            do (rotatef (aref heap child) (aref heap parent))))
    top))

(defun bpe-fill-bytes (work byte-ids text start end)
  "Write each UTF-8 byte of TEXT[START, END) and its token id into WORK; return the count."
  (declare (type text text) (fixnum start end) (type (simple-array (unsigned-byte 32) (256)) byte-ids))
  (bpe-work-reserve work (* 4 (- end start)))
  (let ((bytes (bpe-work-bytes work)) (ids (bpe-work-ids work)) (n 0))
    (declare (fixnum n))
    (flet ((emit (byte) (setf (aref bytes n) byte (aref ids n) (aref byte-ids byte)) (incf n)))
      (declare (inline emit))
      (loop for i of-type fixnum from start below end
            for code of-type fixnum = (char-code (schar text i))
            do (cond ((< code #x80) (emit code))
                     ((< code #x800) (emit (logior #xC0 (ash code -6)))
                      (emit (logior #x80 (logand code #x3F))))
                     ((< code #x10000) (emit (logior #xE0 (ash code -12)))
                      (emit (logior #x80 (logand (ash code -6) #x3F)))
                      (emit (logior #x80 (logand code #x3F))))
                     (t (emit (logior #xF0 (ash code -18)))
                        (emit (logior #x80 (logand (ash code -12) #x3F)))
                        (emit (logior #x80 (logand (ash code -6) #x3F)))
                        (emit (logior #x80 (logand code #x3F)))))))
    n))

(defun bpe-merge-piece (tokenizer work n out)
  "Merge the N byte ids in WORK by rank and append the result to OUT."
  (declare (fixnum n))
  (let* ((merges (tokenizer-merges tokenizer)) (vocab (length (tokenizer-tokens tokenizer)))
         (ids (bpe-work-ids work)) (next (bpe-work-next work))
         (previous (bpe-work-previous work)) (heap (bpe-work-heap work)))
    (declare (fixnum vocab))
    (setf (fill-pointer heap) 0)
    (dotimes (i n)
      (setf (aref next i) (if (= i (1- n)) -1 (1+ i)) (aref previous i) (1- i)))
    (flet ((pair-value (left)
             (let ((right (aref next left)))
               (and (>= right 0)
                    (gethash (+ (* (aref ids left) vocab) (aref ids right)) merges)))))
      (flet ((offer (left)
               (when (>= left 0)
                 (let ((value (pair-value left)))
                   (when value
                     (heap-push heap (+ (logandc2 (the fixnum value) #xFFFFFFFF) left)))))))
        (dotimes (i (1- n)) (offer i))
        (loop while (plusp (length heap))
              do (let* ((key (heap-pop heap)) (left (ldb (byte 32 0) key))
                        (value (and (>= (aref ids left) 0) (pair-value left))))
                   (when (and value (= (ash (the fixnum value) -32) (ash key -32)))
                     (let ((right (aref next left)))
                       (setf (aref ids left) (ldb (byte 32 0) value)
                             (aref ids right) -1
                             (aref next left) (aref next right))
                       (when (>= (aref next right) 0)
                         (setf (aref previous (aref next right)) left))
                       (offer (aref previous left))
                       (offer left)))))))
    (loop for i of-type fixnum = 0 then (aref next i)
          while (>= i 0)
          do (vector-push-extend (aref ids i) out))))

(defun whole-piece-id (tokenizer work n)
  "The id of the vocabulary entry spelling the N bytes in WORK, or NIL."
  (let ((spelling (make-string n)) (bytes (bpe-work-bytes work)))
    (dotimes (i n)
      (setf (schar spelling i) (aref *byte-chars* (aref bytes i))))
    (values (gethash spelling (tokenizer-ids tokenizer)))))

(defun encode-plain (tokenizer text out work)
  "Pre-tokenize and merge TEXT, appending token ids to OUT."
  (let ((byte-ids (tokenizer-byte-ids tokenizer)) (ignore-merges (tokenizer-ignore-merges-p tokenizer)))
    (map-pre-pieces (lambda (start end)
                      (let* ((n (bpe-fill-bytes work byte-ids text start end))
                             (whole (and ignore-merges (whole-piece-id tokenizer work n))))
                        (if whole
                            (vector-push-extend whole out)
                            (bpe-merge-piece tokenizer work n out))))
                    (tokenizer-matchers tokenizer) text)))

(defun special-match (tokenizer text position special)
  "The id of the special token TEXT has at POSITION, or NIL. Control and unknown tokens match
only when SPECIAL; user-defined tokens always do."
  (declare (type text text) (fixnum position))
  (loop for (token . id) in (gethash (schar text position) (tokenizer-specials tokenizer))
        for end of-type fixnum = (+ position (length (the string token)))
        when (and (<= end (length text))
                  (or special (= (aref (tokenizer-types tokenizer) id) +token-user-defined+))
                  (string= token text :start2 position :end2 end))
          return (values id end)))

(defun special-id (tokenizer key)
  (getf (tokenizer-special-ids tokenizer) key))

(defgeneric normalize-tokenizer-text (tokenizer text)
  (:method (tokenizer text) (declare (ignore tokenizer)) text))

(defun encode (tokenizer text &key special (add-bos (tokenizer-add-bos-p tokenizer)))
  "Token ids for TEXT. ADD-BOS defaults to the file's add_bos_token. SPECIAL also matches
control and unknown tokens in TEXT; user-defined tokens always match."
  (let ((text (coerce-text (normalize-tokenizer-text tokenizer text))) (work (make-bpe-work))
        (out (make-array 0 :element-type '(unsigned-byte 32) :adjustable t :fill-pointer 0))
        (segment 0) (position 0))
    (when add-bos
      (vector-push-extend (or (special-id tokenizer :bos)
                              (tokenizer-fail 'tokenizer-error "Tokenizer has no BOS token"))
                          out))
    (flet ((flush (end)
             (when (< segment end)
               (encode-plain tokenizer (if (and (zerop segment) (= end (length text)))
                                           text
                                           (subseq text segment end))
                             out work))))
      (loop while (< position (length text))
            do (multiple-value-bind (id end) (special-match tokenizer text position special)
                 (cond (id (flush position)
                           (vector-push-extend id out)
                           (setf position end segment end))
                       (t (incf position)))))
      (flush (length text)))
    (coerce out '(simple-array (unsigned-byte 32) (*)))))

;;; Decoding

(defun tokenizer-vocab-size (tokenizer)
  (length (tokenizer-tokens tokenizer)))

(defun check-token-id (tokenizer id)
  (unless (and (integerp id) (< -1 id (tokenizer-vocab-size tokenizer)))
    (tokenizer-fail 'tokenizer-error "Invalid token id ~S" id)))

(defun token-id (tokenizer string)
  "The id of the vocabulary entry STRING, or NIL."
  (values (gethash string (tokenizer-ids tokenizer))))

(defun token-bytes (tokenizer id)
  "The bytes token ID stands for, as a (simple-array (unsigned-byte 8) (*))."
  (check-token-id tokenizer id)
  (svref (tokenizer-octets tokenizer) id))

(defun tokenizer-special-p (tokenizer id)
  "True for control, unknown and user-defined tokens."
  (check-token-id tokenizer id)
  (<= +token-unknown+ (aref (tokenizer-types tokenizer) id) +token-user-defined+))

(defun tokenizer-special-id (tokenizer key)
  "The id the file declares for KEY (:bos :eos :eot :eom :pad :unk :sep), or NIL."
  (special-id tokenizer key))

(defun token-rendered-p (tokenizer id special)
  "Control and unknown tokens render only when SPECIAL; unused ones never do."
  (let ((type (aref (tokenizer-types tokenizer) id)))
    (cond ((member type '(0 5)) nil)
          ((or (= type +token-unknown+) (= type +token-control+)) special)
          (t t))))

(defun octets-string (octets &key (start 0) (end (length octets)))
  (babel:octets-to-string octets :encoding :utf-8 :start start :end end :errorp nil))

(defun decode (tokenizer ids &key special)
  "The text of token IDS. Control and unknown tokens are left out unless SPECIAL."
  (let ((out (make-array 0 :element-type '(unsigned-byte 8) :adjustable t :fill-pointer 0)))
    (map nil (lambda (id)
               (check-token-id tokenizer id)
               (when (token-rendered-p tokenizer id special)
                 (loop for byte across (svref (tokenizer-octets tokenizer) id)
                       do (vector-push-extend byte out))))
         ids)
    (octets-string out)))

(defstruct (detokenizer (:constructor %make-detokenizer (tokenizer special)))
  tokenizer special
  (pending (make-array 8 :element-type '(unsigned-byte 8) :adjustable t :fill-pointer 0)))

(defun make-detokenizer (tokenizer &key special)
  "A stateful decoder for streaming: feed ids to DETOKENIZER-PUSH."
  (%make-detokenizer tokenizer special))

(defun incomplete-tail-start (octets)
  "Index where a UTF-8 sequence cut off by the end of OCTETS begins, or the length when none."
  (let ((n (length octets)))
    (loop for i from (1- n) downto (max 0 (- n 3))
          for byte = (aref octets i)
          do (cond ((= (logand byte #xC0) #x80))
                   (t (return (if (> (cond ((= (logand byte #xE0) #xC0) 2)
                                           ((= (logand byte #xF0) #xE0) 3)
                                           ((= (logand byte #xF8) #xF0) 4)
                                           (t 1))
                                     (- n i))
                                  i
                                  n))))
          finally (return n))))

(defun detokenizer-push (detokenizer id)
  "Feed token ID; returns the text it completes, which is \"\" while a character is unfinished."
  (let ((tokenizer (detokenizer-tokenizer detokenizer)) (pending (detokenizer-pending detokenizer)))
    (check-token-id tokenizer id)
    (when (token-rendered-p tokenizer id (detokenizer-special detokenizer))
      (loop for byte across (svref (tokenizer-octets tokenizer) id)
            do (vector-push-extend byte pending)))
    (let ((cut (incomplete-tail-start pending)))
      (prog1 (octets-string pending :end cut)
        (replace pending pending :start2 cut)
        (setf (fill-pointer pending) (- (length pending) cut))))))

(defun detokenizer-finish (detokenizer)
  "The text of any unfinished character, as U+FFFD."
  (let ((pending (detokenizer-pending detokenizer)))
    (prog1 (octets-string pending)
      (setf (fill-pointer pending) 0))))
