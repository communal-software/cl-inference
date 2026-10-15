(in-package #:cl-inference/local)

;;; Sessions

(defconstant +default-context-cap+ 4096
  "The n-ctx a session gets when the model declares none, and the most it takes from one that does.")

(defstruct (session (:constructor %make-session
                       (model tokenizer chat-template n-ctx n-batch cache cache-type reuse-prefix)))
  model tokenizer chat-template n-ctx n-batch cache cache-type reuse-prefix
  (position 0)
  (tokens (make-array 0 :element-type 'integer :adjustable t :fill-pointer 0)))

(defun default-n-ctx (model)
  (let ((length (hparam-or (model-architecture model) 'context-length nil)))
    (if length (min length +default-context-cap+) +default-context-cap+)))

(defun make-session (model &key tokenizer chat-template n-ctx (n-batch 512)
                               (cache-type :f32) (reuse-prefix t))
  "A session over MODEL. N-CTX is the most positions it holds: the model's context-length hparam
up to 4096 by default. N-BATCH is the most tokens one prefill pass feeds.
CACHE-TYPE is :F32 or :Q8; REUSE-PREFIX controls complete-prompt matching."
  (let ((n-ctx (or n-ctx (default-n-ctx model))))
    (unless (and (integerp n-ctx) (plusp n-ctx) (integerp n-batch) (plusp n-batch))
      (session-fail "n-ctx and n-batch must be positive integers"))
    (unless (member cache-type '(:f32 :q8))
      (session-fail "cache-type must be :f32 or :q8"))
    (%make-session model tokenizer chat-template n-ctx n-batch
                   (make-model-cache model n-ctx :cache-type cache-type) cache-type reuse-prefix)))

(defun close-session (session)
  "Release a session's execution cache."
  (close-cache (session-cache session))
  (session-reset session)
  nil)

(defun session-reset (session)
  "Forget every position. The cache buffers are kept."
  (setf (session-position session) 0
        (fill-pointer (session-tokens session)) 0)
  session)

(defun session-prefill (session tokens &key check)
  "Feed TOKENS after the current position, N-BATCH at a time. Returns (values logits failure):
the last position's logits, or NIL and a failed result when CHECK, called before each pass,
returns one. Signals CONTEXT-OVERFLOW when TOKENS do not fit."
  (let* ((tokens (coerce tokens 'vector)) (count (length tokens)) (logits nil)
         (n-batch (session-n-batch session)))
    (when (> (+ (session-position session) count) (session-n-ctx session))
      (context-fail "~D tokens after position ~D do not fit in a context of ~D"
                    count (session-position session) (session-n-ctx session)))
    (loop for offset from 0 below count by n-batch
          for failure = (and check (funcall check))
          when failure do (return-from session-prefill (values nil failure))
          do (let ((batch (subseq tokens offset (min count (+ offset n-batch))))
                   (committed nil))
               (unwind-protect
                    (progn
                      (setf logits (model-forward (session-model session) batch
                                                  :start (session-position session)
                                                  :cache (session-cache session)))
                      (map nil (lambda (token) (vector-push-extend token (session-tokens session))) batch)
                      (incf (session-position session) (length batch))
                      (setf committed t))
                 (unless committed (session-reset session)))))
    (values logits nil)))

(defun prepare-prompt (session tokens reuse-prefix)
  (let ((count (length tokens)) (matched 0) (history (session-tokens session)))
    (when (> count (session-n-ctx session))
      (context-fail "~D prompt tokens do not fit in a context of ~D" count (session-n-ctx session)))
    (when reuse-prefix
      (loop while (and (< matched (min (length history) (1- count)))
                       (= (aref history matched) (elt tokens matched)))
            do (incf matched)))
    (setf (session-position session) matched
          (fill-pointer history) matched)
    matched))

;;; Sampling and stop strings

(declaim (ftype function sample))

(defun greedy-sampler (logits)
  "The id of the largest of LOGITS, shape (1 vocab)."
  (ct:tref (ct:argmax logits :axis 1) 0))

(defun stop-split (pending stops)
  "Split PENDING text against the STOPS strings. Returns (values release held matched): the text
safe to emit, a tail that may begin a stop, and whether a stop occurred, in which case RELEASE
is the text before it."
  (let ((hit nil) (length (length pending)) (keep 0))
    (dolist (stop stops)
      (let ((at (search stop pending)))
        (when (and at (or (null hit) (< at hit))) (setf hit at))))
    (if hit
        (values (subseq pending 0 hit) "" t)
        (progn
          (dolist (stop stops)
            (loop for k from (min (1- (length stop)) length) downto (1+ keep)
                  when (string= stop pending :end1 k :start2 (- length k))
                    do (setf keep k) (return)))
          (values (subseq pending 0 (- length keep)) (subseq pending (- length keep)) nil)))))

;;; Turns

(defun turn-check (timeout cancel)
  "A function returning the failed result once TIMEOUT milliseconds from now pass or CANCEL fires."
  (let ((deadline (and timeout (+ (get-internal-real-time)
                                  (round (* timeout internal-time-units-per-second) 1000)))))
    (lambda ()
      (cond ((and cancel (ci:cancelled-p cancel)) (ci:fail :cancelled))
            ((and deadline (>= (get-internal-real-time) deadline)) (ci:fail :timeout))))))

(defun messages-p (input)
  (and (consp input) (consp (first input))))

(defun prompt-tokens (session input)
  "INPUT's token ids, rendering messages through the session's template."
  (cond ((messages-p input)
         (unless (and (session-chat-template session) (session-tokenizer session))
           (session-fail "Messages need a session with a chat template and a tokenizer"))
         (encode-chat (session-chat-template session) (session-tokenizer session) input))
        (t input)))

(defun default-stop-ids (session)
  (and (session-chat-template session) (session-tokenizer session)
       (chat-template-stop-ids (session-chat-template session) (session-tokenizer session))))

(defun generate (session input &key (sampler #'greedy-sampler) max-tokens stop
                                 (stop-ids (default-stop-ids session))
                                 (timeout ci:+default-timeout+) cancel sink ref special prompt-p
                                 (reuse-prefix (session-reuse-prefix session)))
  "Run one turn: feed INPUT, then sample until a stop id, a STOP string, MAX-TOKENS or a full
context. Token ids append unless PROMPT-P; complete prompts and messages reuse matching prefixes.
SAMPLER is a sampler object or a function from the logits to an id. TIMEOUT (milliseconds, covering the whole turn) and CANCEL end
the turn early. SINK receives client events: text deltas, then one :done.
Returns (values text finish-reason token-ids usage). FINISH-REASON is :stop, :length or a
failed result such as (:error :timeout). TEXT is NIL without a tokenizer. SPECIAL retains control
tokens in decoded output; stop ids still end the turn without appearing in TEXT."
  (let ((tokenizer (session-tokenizer session)))
    (when (and stop (null tokenizer)) (session-fail "Stop strings need a tokenizer"))
    (dolist (string stop)
      (unless (and (stringp string) (plusp (length string)))
        (session-fail "A stop string must be a non-empty string, not ~S" string)))
    (let* ((check (turn-check timeout cancel))
           (prompt (coerce (prompt-tokens session input) 'vector))
           (detokenizer (and tokenizer (make-detokenizer tokenizer :special special)))
           (text (make-string-output-stream))
           (held "") (ids nil) (count 0) (logits nil) (finish nil) (matched nil) (cached 0))
      (when (zerop (length prompt)) (session-fail "Nothing to generate from"))
      (setf finish (funcall check))
      (unless finish
        (when (or prompt-p (messages-p input))
          (setf cached (prepare-prompt session prompt reuse-prefix))))
      (flet ((release (string)
               (when (plusp (length string))
                 (write-string string text)
                 (ci:emit-event sink (ci:text-delta ref string)))))
        (unless finish
          (multiple-value-setq (logits finish)
            (session-prefill session (subseq prompt cached) :check check)))
        (loop until finish
              do (setf finish (or (funcall check)
                                  (and max-tokens (>= count max-tokens) :length)
                                  (and (>= (session-position session) (session-n-ctx session))
                                       :length)))
                 (unless finish
                   (let ((token (sample sampler logits)))
                     (if (member token stop-ids)
                         (setf finish :stop)
                         (progn
                           (push token ids)
                           (incf count)
                           (when detokenizer
                             (multiple-value-bind (out keep hit)
                                 (stop-split (concatenate 'string held
                                                          (detokenizer-push detokenizer token))
                                             stop)
                               (release out)
                               (setf held keep matched hit finish (and hit :stop))))
                           (setf logits (session-prefill session (list token))))))))
        (when (and detokenizer (not matched))
          (release (concatenate 'string held (detokenizer-finish detokenizer)))))
      (ci:emit-event sink (ci:done ref finish))
      (values (and detokenizer (get-output-stream-string text))
              finish
              (nreverse ids)
              (list :prompt-tokens (length prompt) :cached-prompt-tokens cached :completion-tokens count
                    :total-tokens (+ (length prompt) count))))))
