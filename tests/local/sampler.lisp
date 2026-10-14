(in-package #:cl-inference/local/tests)

(in-suite :cl-inference/local)

(defun logits-of (&rest values)
  (ct:from-data (list (mapcar (lambda (x) (coerce x 'single-float)) values))))

(defun log-logits (&rest probabilities)
  (apply #'logits-of (mapcar #'log probabilities)))

(defun draws (sampler logits count)
  (loop repeat count collect (local:sample sampler logits)))

(defun logits-with (bits)
  (let ((logits (logits-of 1 2 3)))
    (float-features:with-float-traps-masked (:invalid :overflow :divide-by-zero)
      (setf (ct:tref logits 0 1) (float-features:bits-single-float bits)))
    logits))

;;; Selection

(test sampler-with-zero-temperature-is-greedy
  (let ((logits (logits-of 1 5 2 4)))
    (dolist (seed '(0 1 99))
      (is (equal '(1 1 1) (draws (local:make-sampler :temperature 0 :seed seed) logits 3))))))

(test sampler-with-top-k-one-is-greedy
  (let ((logits (logits-of 1 5 2 4)))
    (dolist (seed '(0 1 99))
      (is (equal '(1 1 1 1 1) (draws (local:make-sampler :top-k 1 :seed seed) logits 5))))))

(test sampler-top-k-never-leaves-the-k-largest
  (let ((ids (remove-duplicates (draws (local:make-sampler :top-k 2 :seed 3)
                                       (logits-of 1 5 2 4 0) 200))))
    (is (equal '(1 3) (sort (copy-list ids) #'<)))))

(test sampler-top-p-drops-the-tail
  (let ((logits (log-logits 0.6 0.3 0.1)))
    (is (equal '(0) (remove-duplicates (draws (local:make-sampler :top-p 0.5 :seed 1) logits 50))))
    (is (equal '(0 1) (sort (remove-duplicates
                             (draws (local:make-sampler :top-p 0.85 :seed 1) logits 300))
                            #'<)))))

(test sampler-tiny-top-p-keeps-the-largest
  (is (equal '(2 2 2) (draws (local:make-sampler :top-p 1d-9 :seed 5) (logits-of 1 2 9) 3))))

(test sampler-breaks-ties-by-id
  (is (equal '(0 0 0) (draws (local:make-sampler :top-k 1 :seed 8) (logits-of 2 2 2) 3))))

(test sampler-draws-from-the-softmax
  (let* ((draws (draws (local:make-sampler :seed 7) (log-logits 0.5 0.3 0.2) 4000))
         (counts (loop for id below 3 collect (/ (count id draws) 4000.0))))
    (is (every (lambda (got want) (< (abs (- got want)) 0.03)) counts '(0.5 0.3 0.2)))))

(test sampler-temperature-sharpens-and-flattens
  (let ((logits (logits-of 1 2)))
    (is (> (count 1 (draws (local:make-sampler :temperature 0.1 :seed 2) logits 400))
           (count 1 (draws (local:make-sampler :temperature 1 :seed 2) logits 400))
           (count 1 (draws (local:make-sampler :temperature 10 :seed 2) logits 400))))))

(test sampler-survives-a-denormal-temperature
  (is (equal '(1 1) (draws (local:make-sampler :temperature 1d-310 :seed 1) (logits-of 1 9) 2))))

;;; Seeds

(test sampler-seed-fixes-the-stream
  (let ((logits (log-logits 0.25 0.25 0.25 0.25)))
    (is (equal (draws (local:make-sampler :seed 42) logits 20)
               (draws (local:make-sampler :seed 42) logits 20)))
    (is (not (equal (draws (local:make-sampler :seed 42) logits 20)
                    (draws (local:make-sampler :seed 43) logits 20))))))

(test sampler-reuse-continues-the-stream
  (let ((logits (log-logits 0.25 0.25 0.25 0.25))
        (sampler (local:make-sampler :seed 42)))
    (is (equal (draws (local:make-sampler :seed 42) logits 20)
               (append (draws sampler logits 10) (draws sampler logits 10))))))

(test sampler-seed-has-a-golden-sequence
  (is (equal '(2 0 1 1 0 3 0 3 1 2 0 1)
             (draws (local:make-sampler :seed 42) (log-logits 0.25 0.25 0.25 0.25) 12))))

(test splitmix64-matches-integer-reference-vectors
  (dolist (case '((0 #xE220A8397B1DCDAF #x9E3779B97F4A7C15)
                  (42 #xBDD732262FEB6E95 #x9E3779B97F4A7C3F)
                  (#x9E3779B97F4A7C3F #x28EFE333B266F103 #x3C6EF372FE94F854)
                  (#xFFFFFFFFFFFFFFFF #xE4D971771B652C20 #x9E3779B97F4A7C14)
                  (#x61C8864680B583EB 0 0)))
    (destructuring-bind (seed expected-output expected-state) case
      (multiple-value-bind (output state) (local::splitmix-next seed)
        (is (= expected-output output))
        (is (= expected-state state))))))

(test sampler-uniform-draws-use-the-upper-53-bits
  (dolist (seed '(0 42 #xFFFFFFFFFFFFFFFF #x61C8864680B583EB))
    (let ((sampler (local:make-sampler :seed seed)))
      (multiple-value-bind (output state) (local::splitmix-next seed)
        (let ((draw (local::next-uniform sampler)))
          (is (typep draw 'double-float))
          (is (<= 0d0 draw))
          (is (< draw 1d0))
          (is (= (/ (ash output -11) (expt 2 53)) draw))
          (is (= state (local::sampler-state sampler))))))))

(test sampler-without-a-seed-reports-the-one-it-drew
  (let ((a (local:make-sampler)) (logits (log-logits 0.25 0.25 0.25 0.25)))
    (is (integerp (local:sampler-seed a)))
    (is (equal (draws a logits 10)
               (draws (local:make-sampler :seed (local:sampler-seed a)) logits 10)))))

;;; Arguments

(test make-sampler-rejects-bad-arguments
  (dolist (args '((:temperature -1) (:temperature :hot) (:top-k 0) (:top-k 1.5)
                  (:top-p 0) (:top-p 1.1) (:seed 1.5)))
    (signals local:sampler-error (apply #'local:make-sampler args))))

(test sampler-error-is-a-session-error
  (signals local:session-error (local:make-sampler :top-k 0)))

;;; Non-finite logits

(test sampling-signals-on-non-finite-logits
  (dolist (bits '(#x7FC00000 #x7F800000 #xFF800000))
    (let ((logits (logits-with bits)))
      (signals local:sampler-error (local:sample (local:make-sampler :seed 1) logits))
      (signals local:sampler-error (local:sample (local:make-sampler :temperature 0) logits))
      (signals local:sampler-error (local:sample #'local:greedy-sampler logits))
      (signals local:sampler-error (local:sample (lambda (logits) (declare (ignore logits)) 0)
                                                 logits)))))

;;; In a turn

(test generate-takes-a-sampler-object
  (flet ((turn ()
           (with-session (session model) ()
             (local:generate session '(3 0 5) :max-tokens 6
                                              :sampler (local:make-sampler :temperature 0.8
                                                                           :seed 1)))))
    (let ((first (nth-value 2 (turn))))
      (is (= 6 (length first)))
      (is (equal first (nth-value 2 (turn)))))))
