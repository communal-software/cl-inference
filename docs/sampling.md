# Sampling

A sampler picks the next token id from the logits. `generate` takes one as `:sampler`; the default is `greedy-sampler`.

```lisp
(local:generate session messages
                :sampler (local:make-sampler :temperature 0.7 :top-k 40 :top-p 0.9 :seed 42))
```

## Samplers

| Sampler | Does |
|---|---|
| `greedy-sampler` | The largest logit's id |
| `(make-sampler &key temperature top-k top-p seed)` | Temperature, top-k and top-p, then a seeded draw |
| Any function | Called with the `(1 vocab)` logits; returns an id |

`(sample sampler logits)` is the generic behind all three.

## Options

| Option | Default | Meaning |
|---|---|---|
| `:temperature` | 1 | Logits are divided by it; 0 is greedy |
| `:top-k` | `NIL` | Keep the `k` largest logits |
| `:top-p` | 1 | Keep the smallest set whose probability reaches `p` (always at least one token) |
| `:seed` | random | Integer seed; `sampler-seed` returns the one in use |

Filters apply in order: temperature, top-k, top-p. Equal logits rank by lower id first.

## Seeds

The sampler holds its random stream. Reusing it continues the stream; a new sampler with the same seed repeats it.[^prng]

```lisp
(let ((a (local:make-sampler :seed 7)) (b (local:make-sampler :seed 7)))
  (equal (loop repeat 5 collect (local:sample a logits))
         (loop repeat 5 collect (local:sample b logits))))   ; => T
```

## Errors

| Condition | Signalled when |
|---|---|
| `sampler-error` | An option is out of range, or the logits hold a NaN or infinity |

`sampler-error` is a `session-error`. Every call to `sample` checks the logits first, whatever the sampler, so a function sampler is checked too.[^traps]

## Limitations

| Missing | Tracker |
|---|---|
| Seeded golden sequence differs on CCL arm64 | [#46](https://github.com/communal-software/cl-inference/issues/46) |

[^prng]: The stream is splitmix64, so a seed repeats its stream; CCL has a [portability limitation](#limitations).
[^traps]: SBCL traps floating-point invalid operations by default, so a NaN produced during the forward pass signals there. The check covers values that arrive without trapping, such as NaN weights read from a file.
