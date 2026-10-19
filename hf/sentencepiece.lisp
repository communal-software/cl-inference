(in-package #:cl-inference/hf)

(defclass hf-sentencepiece-bpe (local:byte-level-bpe) ()
  (:metaclass local:tokenizer-class))

(defclass hf-gemma-bpe (hf-sentencepiece-bpe) ()
  (:metaclass local:tokenizer-class))

(defparameter *hf-sentencepiece-pipeline*
  (parse-json-object
   (local::utf8-octets
    "{
      \"normalizer\": {\"type\": \"Sequence\", \"normalizers\": [
        {\"type\": \"Prepend\", \"prepend\": \"▁\"},
        {\"type\": \"Replace\", \"pattern\": {\"String\": \" \"}, \"content\": \"▁\"}]},
      \"decoder\": {\"type\": \"Sequence\", \"decoders\": [
        {\"type\": \"Replace\", \"pattern\": {\"String\": \"▁\"}, \"content\": \" \"},
        {\"type\": \"ByteFallback\"}, {\"type\": \"Fuse\"},
        {\"type\": \"Strip\", \"content\": \" \", \"start\": 1, \"stop\": 0}]},
      \"post_processor\": {\"type\": \"TemplateProcessing\",
        \"single\": [{\"SpecialToken\": {\"id\": \"<s>\", \"type_id\": 0}},
                   {\"Sequence\": {\"id\": \"A\", \"type_id\": 0}}],
        \"pair\": [{\"SpecialToken\": {\"id\": \"<s>\", \"type_id\": 0}},
                 {\"Sequence\": {\"id\": \"A\", \"type_id\": 0}},
                 {\"SpecialToken\": {\"id\": \"<s>\", \"type_id\": 1}},
                 {\"Sequence\": {\"id\": \"B\", \"type_id\": 1}}],
        \"special_tokens\": {\"<s>\": {\"id\": \"<s>\", \"ids\": [1], \"tokens\": [\"<s>\"]}}}}")))

(defparameter *hf-gemma-pipeline*
  (let ((pipeline (parse-json-object
                   (local::utf8-octets
                    "{\"normalizer\":{\"type\":\"Replace\",\"pattern\":{\"String\":\" \"},\"content\":\"▁\"},
                      \"pre_tokenizer\":{\"type\":\"Split\",\"pattern\":{\"String\":\" \"},\"behavior\":\"MergedWithPrevious\",\"invert\":false},
                      \"decoder\":{\"type\":\"Sequence\",\"decoders\":[{\"type\":\"Replace\",\"pattern\":{\"String\":\"▁\"},\"content\":\" \"},{\"type\":\"ByteFallback\"},{\"type\":\"Fuse\"}]}}"))))
    (setf (gethash "post_processor" pipeline)
          (parse-json-object
           (local::utf8-octets
            "{\"type\":\"TemplateProcessing\",\"single\":[{\"SpecialToken\":{\"id\":\"<bos>\",\"type_id\":0}},{\"Sequence\":{\"id\":\"A\",\"type_id\":0}}],
              \"pair\":[{\"SpecialToken\":{\"id\":\"<bos>\",\"type_id\":0}},{\"Sequence\":{\"id\":\"A\",\"type_id\":0}},{\"SpecialToken\":{\"id\":\"<bos>\",\"type_id\":1}},{\"Sequence\":{\"id\":\"B\",\"type_id\":1}}],
              \"special_tokens\":{\"<bos>\":{\"id\":\"<bos>\",\"ids\":[2],\"tokens\":[\"<bos>\"]}}}")))
    pipeline))

(defun hf-json-equal-p (left right)
  (cond ((hash-table-p right)
         (and (hash-table-p left) (= (hash-table-count left) (hash-table-count right))
              (loop for key being the hash-keys of right using (hash-value value)
                    always (and (nth-value 1 (gethash key left))
                                (hf-json-equal-p (gethash key left) value)))))
        ((and (vectorp right) (not (stringp right)))
         (and (vectorp left) (not (stringp left)) (= (length left) (length right))
              (every #'hf-json-equal-p left right)))
        (t (equal left right))))

(defun hf-sentencepiece-pipeline-p (object architecture)
  (let ((model (gethash "model" object))
        (pipeline (if (equal architecture "gemma3_text") *hf-gemma-pipeline* *hf-sentencepiece-pipeline*)))
    (unless (and (member architecture '("llama" "gemma3_text") :test #'equal) (hash-table-p model)
                 (eq t (gethash "byte_fallback" model)))
      (return-from hf-sentencepiece-pipeline-p nil))
    (unless (and (equal "BPE" (gethash "type" model))
                 (hf-null-p (gethash "dropout" model))
                 (equal "<unk>" (gethash "unk_token" model))
                 (hf-null-p (gethash "continuing_subword_prefix" model))
                 (hf-null-p (gethash "end_of_word_suffix" model))
                 (eq t (gethash "fuse_unk" model))
                 (not (gethash "ignore_merges" model))
                 (every (lambda (key) (hf-null-p (gethash key object)))
                        '("padding" "truncation"))
                 (every (lambda (key)
                          (if (and (equal key "pre_tokenizer") (hf-null-p (gethash key pipeline)))
                              (hf-null-p (gethash key object))
                              (hf-json-equal-p (gethash key object) (gethash key pipeline))))
                        '("normalizer" "pre_tokenizer" "decoder" "post_processor")))
      (local::tokenizer-fail 'local:tokenizer-error "Unsupported HF SentencePiece BPE pipeline"))
    t))

(defmethod local:tokenizer-load ((tokenizer hf-sentencepiece-bpe) format (weights hf-weights))
  (declare (ignore format))
  (let* ((object (hf-sidecar weights "tokenizer.json" t))
         (config (or (hf-sidecar weights "tokenizer_config.json") (make-hash-table :test 'equal)))
         (gemma (typep tokenizer 'hf-gemma-bpe)))
    (unless (hf-sentencepiece-pipeline-p object (if gemma "gemma3_text" "llama"))
      (local::tokenizer-fail 'local:tokenizer-error "Not an HF SentencePiece BPE pipeline"))
    (dolist (key '("add_bos_token" "add_eos_token" "add_prefix_space" "split_special_tokens"
                   "clean_up_tokenization_spaces"
                   "clean_up_tokenization_spaces_for_bpe_even_though_it_will_corrupt_output"))
      (unless (member (gethash key config) '(nil t))
        (local::tokenizer-fail 'local:tokenizer-error "HF SentencePiece ~A must be a boolean" key)))
    (when (or (gethash "add_prefix_space" config) (gethash "split_special_tokens" config)
              (gethash "add_eos_token" config) (gethash "clean_up_tokenization_spaces" config)
              (gethash "clean_up_tokenization_spaces_for_bpe_even_though_it_will_corrupt_output" config))
      (local::tokenizer-fail 'local:tokenizer-error "Unsupported HF SentencePiece configuration"))
    (multiple-value-bind (tokens types) (hf-vocabulary object)
      (let* ((size (length tokens)) (ids (make-hash-table :test 'equal :size size))
             (octets (make-array size)) (specials (make-hash-table))
             (byte-ids (make-array 256 :element-type '(unsigned-byte 32))))
        (dotimes (id size)
          (let ((token (aref tokens id)))
            (when (or (zerop (length token)) (gethash token ids))
              (local::tokenizer-fail 'local:tokenizer-error "Empty or duplicate HF token ~S" token))
            (setf (gethash token ids) id
                  (aref octets id) (local::utf8-octets (substitute #\Space #\▁ token)))
            (when (<= local::+token-unknown+ (aref types id) local::+token-user-defined+)
              (push (cons token id) (gethash (char token 0) specials)))))
        (dotimes (byte 256)
          (let ((id (gethash (format nil "<0x~2,'0X>" byte) ids)))
            (unless (and id (= (aref types id) local::+token-normal+))
              (local::tokenizer-fail 'local:tokenizer-error "HF SentencePiece needs all 256 ordinary byte tokens"))
            (setf (aref byte-ids byte) id (aref types id) 6
                  (aref octets id) (make-array 1 :element-type '(unsigned-byte 8) :initial-element byte))))
        (maphash (lambda (char entries)
                   (setf (gethash char specials)
                         (sort entries #'> :key (lambda (entry) (length (car entry)))))) specials)
        (setf (slot-value tokenizer 'local::tokens) tokens
              (slot-value tokenizer 'local::types) types
              (slot-value tokenizer 'local::ids) ids
              (slot-value tokenizer 'local::octets) octets
              (slot-value tokenizer 'local::specials) specials
              (slot-value tokenizer 'local::byte-ids) byte-ids)))
    (local::build-merges tokenizer (hf-merges object))
    (multiple-value-bind (bos id) (hf-bos-postprocessor (gethash "post_processor" object))
      (unless (and (= id (or (local:token-id tokenizer bos) -1))
                   (eql (if gemma 3 0) (local:token-id tokenizer "<unk>"))
                   (eql (if gemma 1 2) (local:token-id tokenizer (if gemma "<eos>" "</s>")))
                   (every (lambda (id) (= (aref (local::tokenizer-types tokenizer) id) local::+token-control+))
                          (if gemma '(0 1 2 3) '(0 1 2))))
        (local::tokenizer-fail 'local:tokenizer-error "HF SentencePiece special tokens disagree with the pipeline"))
      (dolist (entry (if gemma
                         '(("bos_token" . "<bos>") ("eos_token" . "<eos>") ("unk_token" . "<unk>") ("pad_token" . "<pad>"))
                         '(("bos_token" . "<s>") ("eos_token" . "</s>") ("unk_token" . "<unk>"))))
        (multiple-value-bind (value present) (gethash (car entry) config)
          (when (and present (not (equal (if (hash-table-p value) (gethash "content" value) value) (cdr entry))))
            (local::tokenizer-fail 'local:tokenizer-error "HF SentencePiece special-token configuration disagrees")))
        (setf (gethash (car entry) config) (cdr entry)))
      (unless (nth-value 1 (gethash "add_bos_token" config))
        (setf (gethash "add_bos_token" config) t)))
    (hf-special-ids tokenizer weights config)
    (setf (aref (local::tokenizer-types tokenizer) (if gemma 3 0)) local::+token-unknown+)
    tokenizer))

(defmethod local:pre-tokenize ((tokenizer hf-sentencepiece-bpe) text)
  (declare (ignore text))
  (local::tokenizer-fail 'local:tokenizer-error "HF SentencePiece BPE has no split-based pre-tokenizer"))

(defmethod local::encode-plain ((tokenizer hf-sentencepiece-bpe) text out work)
  (let ((text (substitute #\▁ #\Space (if (typep tokenizer 'hf-gemma-bpe) text (concatenate 'string "▁" text)))) (count 0))
    (local::bpe-work-reserve work (* 4 (length text)))
    (loop for char across text
          for id = (local:token-id tokenizer (string char))
          do (if id
                 (progn (setf (aref (local::bpe-work-ids work) count) id) (incf count))
                 (loop for byte across (local::utf8-octets (string char))
                       do (setf (aref (local::bpe-work-ids work) count)
                                (aref (local::tokenizer-byte-ids tokenizer) byte))
                          (incf count))))
    (when (plusp count) (local::bpe-merge-piece tokenizer work count out))))

(defstruct (hf-spm-detokenizer (:include local::detokenizer)
                             (:constructor make-hf-spm-detokenizer))
  (initial t))

(defmethod local:make-detokenizer ((tokenizer hf-sentencepiece-bpe) &key special)
  (make-hf-spm-detokenizer :tokenizer tokenizer :special special))

(defun hf-valid-utf8-p (bytes)
  (loop with size = (length bytes) for i from 0 below size
        for first = (aref bytes i)
        for count = (cond ((< first #x80) 0) ((<= #xC2 first #xDF) 1)
                          ((<= #xE0 first #xEF) 2) ((<= #xF0 first #xF4) 3))
        always (and count (< (+ i count) size)
                    (loop for j from (1+ i) to (+ i count)
                          always (<= #x80 (aref bytes j) #xBF))
                    (or (zerop count)
                        (let ((second (aref bytes (1+ i))))
                          (case first (#xE0 (>= second #xA0)) (#xED (< second #xA0))
                                (#xF0 (>= second #x90)) (#xF4 (< second #x90)) (t t)))))
        do (incf i count)))

(defun hf-spm-flush-bytes (decoder)
  ;; HF invalidates the entire byte run, including otherwise valid prefixes.
  (let* ((pending (local::detokenizer-pending decoder))
         (text (if (hf-valid-utf8-p pending) (local::octets-string pending)
                   (make-string (length pending) :initial-element (code-char #xfffd)))))
    (setf (fill-pointer pending) 0)
    text))

(defun hf-spm-strip-initial (decoder text)
  (when (and (not (typep (local::detokenizer-tokenizer decoder) 'hf-gemma-bpe))
             (hf-spm-detokenizer-initial decoder) (plusp (length text)))
    (setf (hf-spm-detokenizer-initial decoder) nil)
    (when (char= (char text 0) #\Space) (setf text (subseq text 1))))
  text)

(defmethod local:detokenizer-push ((decoder hf-spm-detokenizer) id)
  (let ((tokenizer (local::detokenizer-tokenizer decoder)))
    (local::check-token-id tokenizer id)
    (cond ((not (local::token-rendered-p tokenizer id (local::detokenizer-special decoder))) "")
          ((= (aref (local::tokenizer-types tokenizer) id) 6)
           (vector-push-extend (aref (local:token-bytes tokenizer id) 0) (local::detokenizer-pending decoder))
           "")
          (t (hf-spm-strip-initial
              decoder (concatenate 'string (hf-spm-flush-bytes decoder)
                                   (local::octets-string (local:token-bytes tokenizer id))))))))

(defmethod local:detokenizer-finish ((decoder hf-spm-detokenizer))
  (hf-spm-strip-initial decoder (hf-spm-flush-bytes decoder)))

(defmethod local:decode ((tokenizer hf-sentencepiece-bpe) ids &key special)
  (let ((decoder (local:make-detokenizer tokenizer :special special)))
    (with-output-to-string (out)
      (map nil (lambda (id) (write-string (local:detokenizer-push decoder id) out)) ids)
      (write-string (local:detokenizer-finish decoder) out))))

(defmethod local:pre-tokenize ((tokenizer hf-gemma-bpe) text)
  (let ((start 0) (pieces nil))
    (loop for end from 0 below (length text)
          when (char= (char text end) #\Space)
            do (push (subseq text start (1+ end)) pieces) (setf start (1+ end)))
    (when (< start (length text)) (push (subseq text start) pieces))
    (nreverse pieces)))
