(in-package #:cl-inference/local)

;;; Samplers choose the next token id from the logits. GENERATE calls SAMPLE.

(define-condition sampler-error (session-error) ())

(defun sampler-fail (control &rest args)
  (error 'sampler-error :message (apply #'format nil control args)))

(defgeneric sample (sampler logits)
  (:documentation "The token id SAMPLER chooses from LOGITS, shape (1 vocab)."))

(defmethod sample ((sampler function) logits)
  (funcall sampler logits))

(defun logits-vector (logits)
  "LOGITS as a vector of floats."
  (let ((storage (ct:tensor-storage logits)) (offset (ct:tensor-offset logits))
        (count (ct:tensor-size logits)))
    (if (and (typep storage '(simple-array single-float (*))) (ct:contiguous-p logits))
        (subseq storage offset (+ offset count))
        (let ((vector (make-array count)))
          (dotimes (i count vector)
            (setf (aref vector i) (ct:tref logits 0 i)))))))

(defmethod sample :around (sampler logits)
  (loop for x across (logits-vector logits)
        when (or (float-features:float-nan-p x) (float-features:float-infinity-p x))
          do (sampler-fail "The logits hold a non-finite value: ~A" x))
  (call-next-method))

;;; splitmix64

(deftype u64 () '(unsigned-byte 64))

(defun splitmix-next (state)
  "Returns (values output new-state)."
  ;; CCL 1.13 arm64 miscompiles this mixing with u64 declarations.
  (declare (optimize speed))
  (let* ((state (ldb (byte 64 0) (+ state #x9E3779B97F4A7C15)))
         (z state))
    (setf z (ldb (byte 64 0) (* (logxor z (ash z -30)) #xBF58476D1CE4E5B9)))
    (setf z (ldb (byte 64 0) (* (logxor z (ash z -27)) #x94D049BB133111EB)))
    (values (logxor z (ash z -31)) state)))

;;; Sampler objects

(defclass sampler ()
  ((temperature :initarg :temperature :reader sampler-temperature)
   (top-k :initarg :top-k :reader sampler-top-k)
   (top-p :initarg :top-p :reader sampler-top-p)
   (seed :initarg :seed :reader sampler-seed)
   (state :initarg :seed :type u64 :accessor sampler-state)))

(defun make-sampler (&key (temperature 1) top-k (top-p 1) seed)
  "A sampler that scales the logits by 1/TEMPERATURE, keeps the TOP-K largest, then the smallest
set of those with probability at least TOP-P, and draws from it. TEMPERATURE 0 is greedy. A nil
SEED is drawn at random; SAMPLER-SEED returns the seed in use."
  (unless (and (realp temperature) (>= temperature 0))
    (sampler-fail "temperature must be a real of at least 0, not ~S" temperature))
  (unless (or (null top-k) (typep top-k '(integer 1)))
    (sampler-fail "top-k must be nil or a positive integer, not ~S" top-k))
  (unless (and (realp top-p) (< 0 top-p) (<= top-p 1))
    (sampler-fail "top-p must be a real in (0, 1], not ~S" top-p))
  (unless (or (null seed) (integerp seed))
    (sampler-fail "seed must be nil or an integer, not ~S" seed))
  (let ((seed (ldb (byte 64 0) (or seed (random (expt 2 64) (make-random-state t))))))
    (make-instance 'sampler :temperature temperature :top-k top-k :top-p top-p :seed seed)))

(defun next-uniform (sampler)
  "A double-float in [0, 1) from SAMPLER's stream."
  (multiple-value-bind (output state) (splitmix-next (sampler-state sampler))
    (setf (sampler-state sampler) state)
    (* (ash output -11) (scale-float 1d0 -53))))

(defun ranked-ids (scaled)
  "Ids of SCALED, largest value first, equal values by id."
  (let ((ids (make-array (length scaled))))
    (dotimes (i (length ids)) (setf (aref ids i) i))
    ;; TODO: a full sort per token, a partial selection when top-k is small (#40)
    (sort ids (lambda (a b)
                (let ((x (aref scaled a)) (y (aref scaled b)))
                  (or (> x y) (and (= x y) (< a b))))))))

(defun candidate-weights (sampler scaled ids)
  "The unnormalised probabilities of the ids top-k and top-p keep, in IDS order."
  (let* ((temperature (coerce (sampler-temperature sampler) 'double-float))
         (keep (min (length ids) (or (sampler-top-k sampler) (length ids))))
         (best (aref scaled (aref ids 0)))
         (weights (make-array keep :element-type 'double-float)))
    (dotimes (i keep)
      (let ((gap (- (aref scaled (aref ids i)) best)))
        (setf (aref weights i)
              (if (> gap (* -700 temperature)) (exp (/ gap temperature)) 0d0))))
    (let ((limit (* (coerce (sampler-top-p sampler) 'double-float) (reduce #'+ weights)))
          (total 0d0))
      (dotimes (i keep weights)
        (incf total (aref weights i))
        (when (>= total limit) (return (subseq weights 0 (1+ i))))))))

(defmethod sample ((sampler sampler) logits)
  (if (zerop (sampler-temperature sampler))
      (greedy-sampler logits)
      (let* ((scaled (map '(simple-array double-float (*))
                          (lambda (x) (coerce x 'double-float))
                          (logits-vector logits)))
             (ids (ranked-ids scaled))
             (weights (candidate-weights sampler scaled ids))
             (target (* (next-uniform sampler) (reduce #'+ weights)))
             (total 0d0))
        (dotimes (i (length weights) (aref ids (1- (length weights))))
          (incf total (aref weights i))
          (when (< target total) (return (aref ids i)))))))
