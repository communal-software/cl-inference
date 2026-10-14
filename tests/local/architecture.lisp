(in-package #:cl-inference/local/tests)

(in-suite :cl-inference/local)

(local:define-architecture toy-llama ()
  (:hparams (n-layers) (dim) (rope-theta :default 10000.0))
  (:blocks (attn toy-attention) (mlp toy-mlp))
  (:format :gguf
    (:arch "toy-llama")
    (:rope-pairing :adjacent)
    (:hparam n-layers "~a.block_count")
    (:hparam dim "~a.embedding_length")
    (:hparam rope-theta "~a.rope.freq_base")
    (:tensor attn.q "blk.~d.attn_q.weight")
    (:tensor mlp.up "blk.~d.ffn_up.weight")
    (:tensor output "output.weight" :optional t)))

(local:define-architecture toy-qwen3 (toy-llama)
  (:blocks (attn toy-attention :qk-norm toy-rmsnorm))
  (:format :gguf
    (:arch "toy-qwen3")
    (:rope-pairing :neox)
    (:tensor attn.q-norm "blk.~d.attn_q_norm.weight")))

(local:define-architecture toy-bias ()
  (:format :gguf
    (:tensor attn.q "blk.~d.attn_q.bias")
    (:tensor attn.k "blk.~d.attn_k.bias")))

(local:define-architecture toy-biased (toy-llama toy-bias)
  (:format :gguf (:arch "toy-biased")))

(defun toy-gguf (arch &key (metadata '()) tensors)
  (build-gguf :metadata `(("general.architecture" 8 ,arch)
                          (,(format nil "~a.block_count" arch) 4 2)
                          (,(format nil "~a.embedding_length" arch) 4 8)
                          ,@metadata)
              :tensors tensors))

(defmacro with-toy ((weights arch &rest keys) &body body)
  `(with-gguf-file (path (toy-gguf ,arch ,@keys))
     (local:with-weights (,weights path)
       ,@body)))

(test architecture-child-inherits-and-overrides
  (is (equal '(attn mlp) (local:architecture-block-slots 'toy-qwen3)))
  (is (equal '(toy-attention :qk-norm toy-rmsnorm) (local:architecture-block-spec 'toy-qwen3 'attn)))
  (is (equal '(toy-attention) (local:architecture-block-spec 'toy-llama 'attn)))
  (is (equal '(toy-mlp) (local:architecture-block-spec 'toy-qwen3 'mlp)))
  (is (eq :neox (local:architecture-format-option 'toy-qwen3 :gguf :rope-pairing)))
  (is (eq :adjacent (local:architecture-format-option 'toy-llama :gguf :rope-pairing)))
  (is (equal '(:x nil) (multiple-value-list
                        (local:architecture-format-option 'toy-llama :gguf :absent :x))))
  (is (equal '(n-layers dim rope-theta) (mapcar #'car (local:architecture-hparam-specs 'toy-qwen3)))))

(test architecture-multiple-parents-follow-precedence
  (with-toy (w "toy-biased")
    (let ((arch (local:load-architecture w)))
      (is (equal "blk.1.attn_q.weight" (local:architecture-tensor-name arch 'attn.q 1)))
      (is (equal "blk.1.attn_k.bias" (local:architecture-tensor-name arch 'attn.k 1)))
      (is (equal "blk.1.ffn_up.weight" (local:architecture-tensor-name arch 'mlp.up 1))))))

(test architecture-loads-hparams-and-tensor-names
  (with-toy (w "toy-qwen3" :metadata '(("toy-qwen3.rope.freq_base" 6 1000000f0)))
    (let ((arch (local:load-architecture w)))
      (is (typep arch 'toy-qwen3))
      (is (eq :gguf (local:architecture-format arch)))
      (is (= 2 (local:hparam arch 'n-layers)))
      (is (= 8 (local:hparam arch 'dim)))
      (is (= 1000000f0 (local:hparam arch 'rope-theta)))
      (is (equal "blk.3.attn_q_norm.weight" (local:architecture-tensor-name arch 'attn.q-norm 3)))
      (is (equal "output.weight" (local:architecture-tensor-name arch 'output)))
      (signals local:architecture-error (local:hparam arch 'absent))
      (signals local:architecture-error (local:architecture-tensor-name arch 'absent 0)))))

(test architecture-hparam-default
  (with-toy (w "toy-llama")
    (is (= 10000.0 (local:hparam (local:load-architecture w) 'rope-theta)))))

(test architecture-missing-required-hparam
  (with-gguf-file (path (build-gguf :metadata '(("general.architecture" 8 "toy-llama")
                                                ("toy-llama.block_count" 4 2))))
    (local:with-weights (w path)
      (signals local:architecture-error (local:load-architecture w)))))

(test architecture-unknown
  (with-toy (w "no-such-arch")
    (signals local:unknown-architecture (local:load-architecture w)))
  (with-gguf-file (path (build-gguf :metadata '(("general.name" 8 "x"))))
    (local:with-weights (w path)
      (signals local:unknown-architecture (local:load-architecture w)))))

(test architecture-optional-tensor
  (let ((tensors `(("blk.0.attn_q.weight" (2) 0 ,(f32-octets '(1f0 2f0))))))
    (with-toy (w "toy-llama" :tensors tensors)
      (let ((arch (local:load-architecture w)))
        (is (= 2 (ct:tref (local:architecture-tensor arch w 'attn.q 0) 1)))
        (is (null (local:architecture-tensor arch w 'output)))
        (signals local:weights-error (local:architecture-tensor arch w 'mlp.up 0))))))

(local:define-architecture toy-heads ()
  (:model-blocks (token toy-embedding) (head toy-head))
  (:format :gguf
    (:arch "toy-heads")
    (:tensor token.embd "token_embd.weight")
    (:tensor head.output "output.weight" :fallback token.embd)))

(local:define-architecture toy-heads-child (toy-heads)
  (:model-blocks (head toy-other-head)))

(test architecture-model-blocks-merge-by-name
  (is (equal '(token head) (local:architecture-model-block-slots 'toy-heads-child)))
  (is (equal '(toy-other-head) (local:architecture-model-block-spec 'toy-heads-child 'head)))
  (is (equal '(toy-head) (local:architecture-model-block-spec 'toy-heads 'head)))
  (is (null (local:architecture-block-slots 'toy-heads))))

(test architecture-tensor-falls-back
  (let ((embd `("token_embd.weight" (2) 0 ,(f32-octets '(1f0 2f0)))))
    (with-toy (w "toy-heads" :tensors (list embd))
      (let ((arch (local:load-architecture w)))
        (is (= 2 (ct:tref (local:architecture-tensor arch w 'head.output) 1)))))
    (with-toy (w "toy-heads"
                 :tensors (list embd `("output.weight" (2) 0 ,(f32-octets '(5f0 6f0)))))
      (is (= 6 (ct:tref (local:architecture-tensor (local:load-architecture w) w 'head.output)
                        1))))
    (with-toy (w "toy-heads")
      (signals local:weights-error
        (local:architecture-tensor (local:load-architecture w) w 'head.output)))))

(test architecture-roles-and-hparams-match-by-name
  (with-toy (w "toy-llama")
    (let ((arch (local:load-architecture w)))
      (is (equal "blk.1.attn_q.weight" (local:architecture-tensor-name arch :attn.q 1)))
      (is (= 2 (local:hparam arch :n-layers))))))

(test architecture-ambiguous-match
  (eval '(local:define-architecture toy-twin-a () (:format :gguf (:arch "toy-twin"))))
  (eval '(local:define-architecture toy-twin-b () (:format :gguf (:arch "toy-twin"))))
  (with-gguf-file (path (build-gguf :metadata '(("general.architecture" 8 "toy-twin"))))
    (local:with-weights (w path)
      (signals local:architecture-error (local:load-architecture w)))))

(test architecture-redefinition-propagates-to-children
  (eval '(local:define-architecture toy-base ()
          (:format :gguf (:tensor embed "old.weight"))))
  (eval '(local:define-architecture toy-derived (toy-base)
          (:format :gguf (:arch "toy-derived"))))
  (let ((arch (make-instance 'toy-derived :format :gguf)))
    (is (equal "old.weight" (local:architecture-tensor-name arch 'embed)))
    (eval '(local:define-architecture toy-base ()
            (:format :gguf (:tensor embed "new.weight"))))
    (is (equal "new.weight" (local:architecture-tensor-name arch 'embed)))))

(test architecture-rejects-malformed-definitions
  (dolist (form '((local:define-architecture bad-a () (:unknown))
                  (local:define-architecture bad-b () (:hparams (x :bogus 1)))
                  (local:define-architecture bad-c () (:blocks (just-one)))
                  (local:define-architecture bad-d () (:format "gguf"))
                  (local:define-architecture bad-e () (:format :gguf (:tensor role)))
                  (local:define-architecture bad-f () (:format :gguf (:tensor role "t" :bogus t)))
                  (local:define-architecture bad-g () (:format :gguf (:hparam x 1)))
                  (local:define-architecture bad-i () (:format :gguf (:tensor role "t" :fallback "x")))
                  (local:define-architecture bad-j () (:model-blocks (just-one)))))
    (signals local:architecture-error (macroexpand-1 form)))
  (signals local:architecture-error
    (eval '(local:define-architecture bad-h () (:format :gguf (:hparam undeclared "k"))))))

(local:define-architecture cross-package-child (local:llama)
  (:blocks (attn local:attention :qk-norm t))
  (:format :gguf (:arch "cross-package-child"))
  (:format :safetensors (:arch nil)))

(test child-in-another-package-overrides-an-inherited-block
  (let ((slots (local:architecture-block-slots 'cross-package-child)))
    (is (= 2 (length slots)))
    (is (equal '(local:attention :qk-norm t)
               (local:architecture-block-spec 'cross-package-child 'attn)))))

(local:define-architecture toy-removed (toy-llama toy-heads)
  (:hparams (:remove :dim))
  (:blocks (:remove :mlp) (:remove absent))
  (:model-blocks (:remove :head))
  (:format :gguf
    (:arch "toy-removed")
    (:remove :tensor output)
    (:remove :option :rope-pairing)
    (:remove :hparam rope-theta)))

(test architecture-removes-every-entry-kind
  (is (equal '(n-layers rope-theta)
             (mapcar #'car (local:architecture-hparam-specs 'toy-removed))))
  (is (equal '(attn) (local:architecture-block-slots 'toy-removed)))
  (is (null (local:architecture-block-spec 'toy-removed :mlp)))
  (is (equal '(token) (local:architecture-model-block-slots 'toy-removed)))
  (is (null (local:architecture-model-block-spec 'toy-removed :head)))
  (is (equal '(:fallback nil)
             (multiple-value-list
              (local:architecture-format-option 'toy-removed :gguf :rope-pairing :fallback))))
  (with-toy (w "toy-removed" :metadata '(("toy-removed.rope.freq_base" 6 42f0)))
    (let ((arch (local:load-architecture w)))
      (is (= 2 (local:hparam arch :n-layers)))
      (is (= 10000f0 (local:hparam arch :rope-theta)))
      (signals local:architecture-error (local:hparam arch :dim))
      (signals local:architecture-error (local:architecture-tensor-name arch :output)))))

(local:define-architecture toy-required ()
  (:hparams required)
  (:format :gguf (:arch "toy-required") (:hparam required "required"))
  (:format :safetensors (:hparam required "required_hf")))

(local:define-architecture toy-without-required (toy-required)
  (:hparams (:remove :required))
  (:format :gguf (:arch "toy-without-required")))

(local:define-architecture toy-required-restored (toy-without-required)
  (:hparams (required :default 9)))

(test architecture-removal-cascades-to-all-hparam-mappings
  (dolist (format '(:gguf :safetensors))
    (is (null (local::architecture-hparam-keys
               (find-class 'toy-without-required) format)))
    (is (= 1 (length (local::architecture-hparam-keys
                      (find-class 'toy-required-restored) format)))))
  (with-gguf-file (path (build-gguf :metadata
                                  '(("general.architecture" 8 "toy-without-required"))))
    (local:with-weights (w path)
      (is (typep (local:load-architecture w) 'toy-required-restored))))
  (let ((arch (make-instance 'toy-without-required :required 7)))
    (signals local:architecture-error (local:hparam arch :required))))

(local:define-architecture toy-removal-first (toy-llama)
  (:blocks (:remove attn))
  (:format :gguf (:arch nil)))

(local:define-architecture toy-removal-wins (toy-removal-first toy-bias))
(local:define-architecture toy-attn-parent () (:blocks (attn toy-attention)))
(local:define-architecture toy-declaration-wins (toy-attn-parent toy-removal-first))

(test architecture-removal-follows-parent-precedence
  (is (equal '(mlp) (local:architecture-block-slots 'toy-removal-wins)))
  (is (equal '(mlp attn) (local:architecture-block-slots 'toy-declaration-wins))))

(local:define-architecture toy-ordered-removals (toy-llama)
  (:hparams (extra :default 1) (:remove extra) (extra :default 2))
  (:blocks (:remove attn) (attn toy-other-attention) (:remove mlp))
  (:format :gguf
    (:arch nil)
    (:remove :tensor attn.q)
    (:tensor attn.q "restored.weight")
    (:rope-pairing :neox)
    (:remove :option :rope-pairing))
  (:format :gguf (:rope-pairing :adjacent)))

(test architecture-removals-and-restoration-follow-declaration-order
  (is (equal '(attn) (local:architecture-block-slots 'toy-ordered-removals)))
  (is (equal '(toy-other-attention)
             (local:architecture-block-spec 'toy-ordered-removals :attn)))
  (is (= 2 (getf (cdr (assoc 'extra (local:architecture-hparam-specs 'toy-ordered-removals)))
                :default)))
  (is (eq :adjacent (local:architecture-format-option
                    'toy-ordered-removals :gguf :rope-pairing)))
  (is (equal "restored.weight"
             (local:architecture-tensor-name
              (make-instance 'toy-ordered-removals :format :gguf) :attn.q))))

(test architecture-removal-survives-parent-redefinition
  (eval '(local:define-architecture removal-base () (:blocks (attn old-attention))))
  (eval '(local:define-architecture removal-child (removal-base)
          (:blocks (:remove :attn))))
  (eval '(local:define-architecture removal-base ()
          (:blocks (attn new-attention) (mlp new-mlp))))
  (is (equal '(mlp) (local:architecture-block-slots 'removal-child)))
  (is (equal '(attn mlp) (local:architecture-block-slots 'removal-base))))

(test architecture-rejects-malformed-removals
  (dolist (section '((:hparams (:remove x . y))
                     (:format :gguf (:remove :tensor . role))
                     (:hparams (:remove)) (:hparams (:remove x y))
                     (:blocks (:remove "x")) (:model-blocks (:remove nil))
                     (:format :gguf (:remove :unknown x))
                     (:format :gguf (:remove :tensor))
                     (:format :gguf (:remove :option x))
                     (:format :gguf (:remove :hparam "x"))))
    (signals local:architecture-error
      (macroexpand-1 `(local:define-architecture bad-removal () ,section)))))
