(in-package #:cl-inference/tests)

(in-suite :cl-inference)

(ci:define-architecture toy-llama ()
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

(ci:define-architecture toy-qwen3 (toy-llama)
  (:blocks (attn toy-attention :qk-norm toy-rmsnorm))
  (:format :gguf
    (:arch "toy-qwen3")
    (:rope-pairing :neox)
    (:tensor attn.q-norm "blk.~d.attn_q_norm.weight")))

(ci:define-architecture toy-bias ()
  (:format :gguf
    (:tensor attn.q "blk.~d.attn_q.bias")
    (:tensor attn.k "blk.~d.attn_k.bias")))

(ci:define-architecture toy-biased (toy-llama toy-bias)
  (:format :gguf (:arch "toy-biased")))

(defun toy-gguf (arch &key (metadata '()) tensors)
  (build-gguf :metadata `(("general.architecture" 8 ,arch)
                          (,(format nil "~a.block_count" arch) 4 2)
                          (,(format nil "~a.embedding_length" arch) 4 8)
                          ,@metadata)
              :tensors tensors))

(defmacro with-toy ((weights arch &rest keys) &body body)
  `(with-gguf-file (path (toy-gguf ,arch ,@keys))
     (ci:with-weights (,weights path)
       ,@body)))

(test architecture-child-inherits-and-overrides
  (is (equal '(attn mlp) (ci:architecture-block-slots 'toy-qwen3)))
  (is (equal '(toy-attention :qk-norm toy-rmsnorm) (ci:architecture-block-spec 'toy-qwen3 'attn)))
  (is (equal '(toy-attention) (ci:architecture-block-spec 'toy-llama 'attn)))
  (is (equal '(toy-mlp) (ci:architecture-block-spec 'toy-qwen3 'mlp)))
  (is (eq :neox (ci:architecture-format-option 'toy-qwen3 :gguf :rope-pairing)))
  (is (eq :adjacent (ci:architecture-format-option 'toy-llama :gguf :rope-pairing)))
  (is (equal '(:x nil) (multiple-value-list
                        (ci:architecture-format-option 'toy-llama :gguf :absent :x))))
  (is (equal '(n-layers dim rope-theta) (mapcar #'car (ci:architecture-hparam-specs 'toy-qwen3)))))

(test architecture-multiple-parents-follow-precedence
  (with-toy (w "toy-biased")
    (let ((arch (ci:load-architecture w)))
      (is (equal "blk.1.attn_q.weight" (ci:architecture-tensor-name arch 'attn.q 1)))
      (is (equal "blk.1.attn_k.bias" (ci:architecture-tensor-name arch 'attn.k 1)))
      (is (equal "blk.1.ffn_up.weight" (ci:architecture-tensor-name arch 'mlp.up 1))))))

(test architecture-loads-hparams-and-tensor-names
  (with-toy (w "toy-qwen3" :metadata '(("toy-qwen3.rope.freq_base" 6 1000000f0)))
    (let ((arch (ci:load-architecture w)))
      (is (typep arch 'toy-qwen3))
      (is (eq :gguf (ci:architecture-format arch)))
      (is (= 2 (ci:hparam arch 'n-layers)))
      (is (= 8 (ci:hparam arch 'dim)))
      (is (= 1000000f0 (ci:hparam arch 'rope-theta)))
      (is (equal "blk.3.attn_q_norm.weight" (ci:architecture-tensor-name arch 'attn.q-norm 3)))
      (is (equal "output.weight" (ci:architecture-tensor-name arch 'output)))
      (signals ci:architecture-error (ci:hparam arch 'absent))
      (signals ci:architecture-error (ci:architecture-tensor-name arch 'absent 0)))))

(test architecture-hparam-default
  (with-toy (w "toy-llama")
    (is (= 10000.0 (ci:hparam (ci:load-architecture w) 'rope-theta)))))

(test architecture-missing-required-hparam
  (with-gguf-file (path (build-gguf :metadata '(("general.architecture" 8 "toy-llama")
                                                ("toy-llama.block_count" 4 2))))
    (ci:with-weights (w path)
      (signals ci:architecture-error (ci:load-architecture w)))))

(test architecture-unknown
  (with-toy (w "no-such-arch")
    (signals ci:unknown-architecture (ci:load-architecture w)))
  (with-gguf-file (path (build-gguf :metadata '(("general.name" 8 "x"))))
    (ci:with-weights (w path)
      (signals ci:unknown-architecture (ci:load-architecture w)))))

(test architecture-optional-tensor
  (let ((tensors `(("blk.0.attn_q.weight" (2) 0 ,(f32-octets '(1f0 2f0))))))
    (with-toy (w "toy-llama" :tensors tensors)
      (let ((arch (ci:load-architecture w)))
        (is (= 2 (ct:tref (ci:architecture-tensor arch w 'attn.q 0) 1)))
        (is (null (ci:architecture-tensor arch w 'output)))
        (signals ci:weights-error (ci:architecture-tensor arch w 'mlp.up 0))))))

(test architecture-ambiguous-match
  (eval '(ci:define-architecture toy-twin-a () (:format :gguf (:arch "toy-twin"))))
  (eval '(ci:define-architecture toy-twin-b () (:format :gguf (:arch "toy-twin"))))
  (with-gguf-file (path (build-gguf :metadata '(("general.architecture" 8 "toy-twin"))))
    (ci:with-weights (w path)
      (signals ci:architecture-error (ci:load-architecture w)))))

(test architecture-redefinition-propagates-to-children
  (eval '(ci:define-architecture toy-base ()
          (:format :gguf (:tensor embed "old.weight"))))
  (eval '(ci:define-architecture toy-derived (toy-base)
          (:format :gguf (:arch "toy-derived"))))
  (let ((arch (make-instance 'toy-derived :format :gguf)))
    (is (equal "old.weight" (ci:architecture-tensor-name arch 'embed)))
    (eval '(ci:define-architecture toy-base ()
            (:format :gguf (:tensor embed "new.weight"))))
    (is (equal "new.weight" (ci:architecture-tensor-name arch 'embed)))))

(test architecture-rejects-malformed-definitions
  (dolist (form '((ci:define-architecture bad-a () (:unknown))
                  (ci:define-architecture bad-b () (:hparams (x :bogus 1)))
                  (ci:define-architecture bad-c () (:blocks (just-one)))
                  (ci:define-architecture bad-d () (:format "gguf"))
                  (ci:define-architecture bad-e () (:format :gguf (:tensor role)))
                  (ci:define-architecture bad-f () (:format :gguf (:tensor role "t" :bogus t)))
                  (ci:define-architecture bad-g () (:format :gguf (:hparam x 1)))))
    (signals ci:architecture-error (macroexpand-1 form)))
  (signals ci:architecture-error
    (eval '(ci:define-architecture bad-h () (:format :gguf (:hparam undeclared "k"))))))
