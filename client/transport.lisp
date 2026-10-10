(in-package #:cl-inference/client)

;;; --- Retry-After ------------------------------------------------------

(defparameter +months+
  '("Jan" "Feb" "Mar" "Apr" "May" "Jun" "Jul" "Aug" "Sep" "Oct" "Nov" "Dec"))

(defun http-date-universal-time (text)
  "TEXT, an IMF-fixdate such as \"Wed, 21 Oct 2026 07:28:00 GMT\", as a
universal time, or nil when it is anything else."
  (ignore-errors
   (destructuring-bind (day-name day month-name year clock zone)
       (uiop:split-string text :separator " ")
     (declare (ignore day-name))
     (let ((month (position month-name +months+ :test #'string-equal)))
       (when (and month (string= zone "GMT"))
         (destructuring-bind (hour minute second)
             (mapcar #'parse-integer (uiop:split-string clock :separator ":"))
           (encode-universal-time second minute hour (parse-integer day) (1+ month)
                                  (parse-integer year) 0)))))))

(defun parse-decimal (text)
  "TEXT as a non-negative decimal, or nil."
  (ignore-errors
   (destructuring-bind (whole &optional (fraction "")) (uiop:split-string text :separator ".")
     (and (every #'digit-char-p whole) (every #'digit-char-p fraction)
          (plusp (+ (length whole) (length fraction)))
          (+ (if (plusp (length whole)) (parse-integer whole) 0)
             (if (plusp (length fraction))
                 (/ (parse-integer fraction) (expt 10 (length fraction)))
                 0))))))

(defun retry-after-ms (headers)
  "The wait, in milliseconds, HEADERS ask for, or nil. HEADERS is drakma's
alist. The non-standard retry-after-ms wins over Retry-After, which is
delta-seconds or an HTTP date; a date already past is 0."
  (flet ((header (name) (a:when-let ((value (cdr (assoc name headers))))
                          (string-trim " " value))))
    (or (a:when-let* ((text (header :retry-after-ms))
                      (ms (parse-decimal text)))
          (round ms))
        (a:when-let ((text (header :retry-after)))
          (a:if-let ((seconds (parse-decimal text)))
            (round (* 1000 seconds))
            (a:when-let ((time (http-date-universal-time text)))
              (max 0 (* 1000 (- time (get-universal-time))))))))))

;;; --- connect refusal ---------------------------------------------------

;;; usocket's SBCL :timeout connect path does a non-blocking connect, then
;;; polls GETPEERNAME to detect success. On Linux, a refused connect leaves
;;; GETPEERNAME reporting ENOTCONN indefinitely rather than surfacing
;;; SO_ERROR, so a refusal spins until the whole connect timeout elapses
;;; instead of failing immediately.
;;;
;;; TODO: this binds an unexported usocket symbol to fall back to its legacy
;;; blocking connect, which does fail a refusal immediately (an unreachable
;;; host is still bounded by :timeout). Drop it once usocket's new connect loop
;;; checks SO_ERROR itself (#27).
(defmacro with-immediate-connect-refusal (&body body)
  "Run BODY -- which must make its USOCKET:SOCKET-CONNECT call directly,
inside the same thread -- so a refused connection fails at once rather
than waiting out the connect timeout."
  #+sbcl `(let ((usocket::*socket-connect-nonblock-wait* nil)) ,@body)
  #-sbcl `(progn ,@body))

;;; --- the bounded run ----------------------------------------------------

;;; FUNCTION runs on the calling thread. A watcher thread waits out the
;;; deadline; on a timeout or a cancel it shuts the exchange's socket down,
;;; which wakes a read blocked on it, and interrupts the calling thread out of
;;; FUNCTION with a throw. An exchange is finished under its lock before its
;;; socket is closed, so no interrupt is sent once the cleanup has begun, and
;;; one already in flight finds *EXCHANGE* unbound and does nothing.

(defstruct (exchange (:conc-name exchange-))
  thread socket reason finished (lock (bt:make-lock)) (ended (bt:make-semaphore)))

(defvar *exchange* nil
  "The exchange this thread is running, so an interrupt meant for one that has
finished does nothing.")

(defparameter *sink-grace* 5
  "Seconds a sink has to take the closing :done once the exchange is over.")

;; TODO: one watcher thread per bounded run; a shared timer thread if
;; completions are issued at high rates (#26).
(defun start-watcher (exchange seconds)
  (bt:make-thread (lambda ()
                    (unless (bt:wait-on-semaphore (exchange-ended exchange)
                                                  :timeout (max 0 seconds))
                      (abandon-exchange exchange :timeout)))
                  :name "completion deadline"))

(defun call-bounded (seconds function &key cancel)
  "Call FUNCTION, with a fresh exchange, on this thread, bounded by SECONDS.
Answers (values result reason), REASON being :TIMEOUT or, when CANCEL -- a
cancel token -- fired, :CANCELLED. At either FUNCTION is unwound, rather than
left running until the backend answers or hangs up, and any socket it opened
on the exchange is closed."
  (when (and cancel (cancelled-p cancel))
    (return-from call-bounded (values nil :cancelled)))
  (let ((exchange (make-exchange :thread (bt:current-thread)))
        (result nil))
    (unwind-protect
         (progn
           (start-watcher exchange seconds)
           (when cancel
             (on-cancel cancel (lambda () (abandon-exchange exchange :cancelled))))
           (catch exchange
             (let ((*exchange* exchange))
               (setf result (funcall function exchange)))))
      (finish-exchange exchange)
      (close-socket (exchange-socket exchange))
      (bt:signal-semaphore (exchange-ended exchange)))
    (let ((reason (exchange-reason exchange)))
      (if reason (values nil reason) (values result nil)))))

(defun call-with-deadline (timeout-ms function &key cancel)
  "CALL-BOUNDED for an HTTP exchange. FUNCTION takes CONNECT, a function of a
URL answering a stream ready for drakma's :STREAM."
  (call-bounded (/ timeout-ms 1000)
                (lambda (exchange)
                  (funcall function (lambda (url)
                                      (open-connection url exchange timeout-ms))))
                :cancel cancel))

(defun finish-exchange (exchange)
  "End EXCHANGE, so a cancel or timeout arriving now does nothing."
  (bt:with-lock-held ((exchange-lock exchange))
    (setf (exchange-finished exchange) t)))

(defun close-socket (socket)
  "Shut SOCKET down before closing it: on Linux a close alone leaves a thread
blocked in a read on it asleep, and a shutdown wakes it."
  (when socket
    (ignore-errors (usocket:socket-shutdown socket :io))
    (ignore-errors (usocket:socket-close socket))))

(defun abandon-exchange (exchange reason)
  "End EXCHANGE's run for REASON, :TIMEOUT or :CANCELLED, unless something
already has: shut its socket down, which wakes a read on it, and interrupt
its thread out of FUNCTION. Cheap, and safe to call from any thread."
  (bt:with-lock-held ((exchange-lock exchange))
    (unless (or (exchange-finished exchange) (exchange-reason exchange))
      (setf (exchange-reason exchange) reason)
      (a:when-let ((socket (exchange-socket exchange)))
        (ignore-errors (usocket:socket-shutdown socket :io)))
      (ignore-errors
       (bt:interrupt-thread (exchange-thread exchange)
                            (lambda ()
                              (when (eq *exchange* exchange)
                                (throw exchange nil))))))))

;;; --- the connection -----------------------------------------------------

;;; The connection is opened here rather than left to drakma, so the deadline
;;; has a socket of its own to close: drakma's :connection-timeout only bounds
;;; connecting, not the whole exchange.

(defun header-alist (headers)
  "HEADERS, a coerced plist of names and values, as a lower-cased alist."
  (loop for (name value) on headers by #'cddr
        collect (cons (string-downcase name) value)))

(defun open-connection (url exchange timeout-ms)
  (let* ((uri (puri:parse-uri url))
         (securep (eq (puri:uri-scheme uri) :https))
         ;; An interrupt landing between this call returning and the store
         ;; below leaks the socket (#28).
         (socket (with-immediate-connect-refusal
                   (usocket:socket-connect
                    (puri:uri-host uri) (or (puri:uri-port uri) (if securep 443 80))
                    :element-type '(unsigned-byte 8)
                    ;; Bounds the connect phase alone, ahead of the whole-
                    ;; exchange deadline.
                    :timeout (max 1 (ceiling timeout-ms 1000))
                    :nodelay :if-supported))))
    ;; A cancel that landed while connecting found no socket to close.
    (when (bt:with-lock-held ((exchange-lock exchange))
            (setf (exchange-socket exchange) socket)
            (exchange-reason exchange))
      (close-socket socket)
      (error "cancelled"))
    (wrap-http-stream socket (puri:uri-host uri) securep)))

(defun wrap-http-stream (socket host securep)
  "SOCKET's stream, wrapped exactly as drakma wraps one it opens itself:
chunked framing under a flexi-stream, with SSL attached first when
SECUREP. Passing :stream skips drakma's own wrapping entirely -- it only
adjusts the flexi-stream's element-type and external-format -- so a stream
given raw fails outright, and one without SSL attached sends a TLS
handshake in the clear."
  (let ((raw (usocket:socket-stream socket)))
    (flexi-streams:make-flexi-stream
     (chunga:make-chunked-stream
      (if securep
          (cl+ssl:make-ssl-client-stream raw :hostname host)
          raw))
     ;; Matches drakma's own +LATIN-1+ (specials.lisp), which is internal.
     :external-format (flexi-streams:make-external-format :latin-1 :eol-style :lf))))

(defun character-stream (stream)
  "STREAM as UTF-8 characters. Drakma leaves the stream it was given in the
external format of the response's content type, which is Latin-1 unless the
type names a charset."
  (if (typep stream 'flexi-streams:flexi-stream)
      (progn
        (setf (flexi-streams:flexi-stream-external-format stream)
              (flexi-streams:make-external-format :utf-8 :eol-style :lf)
              (flexi-streams:flexi-stream-element-type stream) 'character)
        stream)
      (flexi-streams:make-flexi-stream stream :external-format :utf-8)))

(defun read-detail (stream)
  "An error response's body, for the detail of a (:backend-error ...)."
  (or (ignore-errors (uiop:slurp-stream-string stream)) ""))

;;; --- the completion -----------------------------------------------------

(defun perform-completion (request opener reader)
  "Run the exchange under the caller's deadline, so a wedged backend costs a
timeout, not a wedged caller. OPENER takes (request connect) and answers
(values stream status headers), HEADERS being drakma's response alist and
optional; READER takes (request stream status) and answers the reply. A
request with a :STREAM sink ends it with exactly one :DONE."
  (let ((sink (getf request :stream)))
    (multiple-value-bind (result reason)
        (call-with-deadline
         (getf request :timeout +default-timeout+)
         (lambda (connect) (attempt-completion request opener reader connect))
         :cancel (getf request :cancel))
      (let ((result (case reason
                      (:timeout (fail :timeout))
                      (:cancelled (fail :cancelled))
                      (t result))))
        (when sink
          ;; Bounded on its own: the exchange's deadline has passed, and a sink
          ;; that blocks on :done must not hold the caller.
          (call-bounded *sink-grace*
                        (lambda (exchange)
                          (declare (ignore exchange))
                          (emit-event sink (done (getf request :ref)
                                                 (done-reason result))))))
        result))))

(defun attempt-completion (request opener reader connect)
  ;; The two failure regions are kept apart: nothing read yet is a transport
  ;; failure, and everything after the status is the backend misbehaving.
  (let (stream status headers)
    (handler-case
        (multiple-value-setq (stream status headers) (funcall opener request connect))
      ;; Nothing was read, so there is no backend answer to report on: a
      ;; refused connection and a peer that hangs up before the status line
      ;; are the same failure to the caller.
      (error () (return-from attempt-completion (fail :unavailable))))
    (unwind-protect
         (handler-case
             (if (<= 200 status 299)
                 (funcall reader request stream status)
                 (backend-error status (read-detail stream)
                                :retry-after (retry-after-ms headers)))
           (error (e) (backend-error status (princ-to-string e))))
      (ignore-errors (close stream)))))
