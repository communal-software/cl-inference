# Qwen3

`cl-inference/local` runs Qwen3 0.6B GGUF models in F16 or Q8_0. `qwen3` inherits from `llama`, with QK normalization, NeoX RoPE and the `qwen3-chat` template. The file's `qwen2` pre-tokenizer selects the existing byte-level BPE tokenizer.

## Generate

Load `cl-inference/gguf` and, for Q8_0, `cl-inference/quant`. The `local` nickname below refers to `cl-inference/local`.

```lisp
(local:with-weights (weights "reference/models/Qwen3-0.6B-Q8_0.gguf")
  (let* ((model (local:load-model weights))
         (tokenizer (local:load-tokenizer weights))
         (template (local:load-chat-template weights))
         (messages '((:role :user :content "What is the capital of France?")))
         (session (local:make-session model :tokenizer tokenizer :chat-template template)))
    (dolist (thinking '(nil t))
      (local:session-reset session)
      (let ((prompt (local:encode-chat template tokenizer messages
                                        :enable-thinking thinking)))
        (multiple-value-bind (text finish ids)
            (local:generate session prompt :max-tokens 64)
          (declare (ignore ids))
          (format t "Thinking ~S (~A): ~A~%" thinking finish text))))))
```

Thinking is enabled by default. Disable it with `:enable-thinking nil` when encoding chat. The reply retains reasoning markers; parsing reasoning and tool calls is a [limitation](limitations.md). A short thinking run can finish with `:length` before producing its answer. End-of-turn tokens produce `:stop` and are omitted from returned text and ids.

## Acceptance inputs

Downloads use immutable revisions and SHA256 pins. Models stay in the gitignored `reference/models/` directory.

| Format | Source revision | SHA256 |
|---|---|---|
| F16 | [second-state/Qwen3-0.6B-GGUF](https://huggingface.co/second-state/Qwen3-0.6B-GGUF/tree/9c1e90ceb7ee4dfe02235b861aa08f0640f89459) | `25fb880334987a71db728fb43af8568a4682208059f754a2735781f5d0970143` |
| Q8_0 | [Qwen/Qwen3-0.6B-GGUF](https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/tree/23749fefcc72300e3a2ad315e1317431b06b590a) | `9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031` |

```sh
mkdir -p reference/models
curl -fL -o reference/models/Qwen3-0.6B-f16.gguf \
  https://huggingface.co/second-state/Qwen3-0.6B-GGUF/resolve/9c1e90ceb7ee4dfe02235b861aa08f0640f89459/Qwen3-0.6B-f16.gguf
curl -fL -o reference/models/Qwen3-0.6B-Q8_0.gguf \
  https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/23749fefcc72300e3a2ad315e1317431b06b590a/Qwen3-0.6B-Q8_0.gguf
shasum -a 256 reference/models/Qwen3-0.6B-*.gguf
```

## Oracle checks

The [acceptance suite](testing.md#acceptance) compares metadata, exact prompt ids, all 28 layer outputs, final logits, ranked top logits and eight greedy tokens. Each format runs plain completion and chat with thinking enabled and disabled. Inputs missing from `reference/` skip individually; all six Qwen cases must run to establish acceptance.

Qwen3 uses the standard [per-format bounds](testing.md#acceptance) with two case-specific exceptions:

| Q8_0 case | Cosine at least | Relative error at most | Top logits |
|---|---|---|---|
| Plain completion | 0.999 | 0.04 | Exact top token and the same unordered top-three set |
| Non-thinking chat | 0.998 | 0.06 | Top three in order |

F16 and Q8_0 thinking chat retain their standard bounds. Prompt and greedy ids match exactly in every case.[^drift]

The measured worst layer cosine/error across the three prompts is `0.99993793` / `0.01114172` for F16 and `0.99843496` / `0.05593544` for Q8_0. All six cases match the oracle's greedy tokens; both non-thinking replies stop after `Paris`, while the eight-token thinking runs finish with `:length`.[^drift]

The development oracle is CPU-only llama.cpp `b11574`. Its local dumper sets `common_chat_templates_inputs.enable_thinking` from `DUMP_THINKING` (`0` disables, `1` enables) before rendering the model's own template.[^oracle]

```sh
for kind in f16 q8; do
  case "$kind" in
    f16) model=reference/models/Qwen3-0.6B-f16.gguf ;;
    q8) model=reference/models/Qwen3-0.6B-Q8_0.gguf ;;
  esac
  reference/llama-oracle.sh "$model" "qwen-$kind" "The capital of France is"
  DUMP_SYSTEM="Answer in one word." DUMP_USER="What is the capital of France?" \
    DUMP_THINKING=1 reference/llama-oracle.sh "$model" "qwen-$kind-thinking" ""
  DUMP_SYSTEM="Answer in one word." DUMP_USER="What is the capital of France?" \
    DUMP_THINKING=0 reference/llama-oracle.sh "$model" "qwen-$kind-chat" ""
done
sbcl --dynamic-space-size 16GB --non-interactive --eval '(require :asdf)' \
  --eval '(asdf:test-system :cl-inference/local/acceptance)'
```

[^oracle]: The oracle, its C++ dumper and dumps stay in gitignored `reference/`. They are development tools, never library dependencies. Lisp compares floating-point traces with the per-format bounds documented in [testing](testing.md#acceptance).

[^drift]: llama.cpp's native Q8_0 matmul also quantizes activations. With identical Q8_0 weights dequantized to F32, Lisp's worst layer error is `0.001359204` on the plain prompt and `0.002398626` on non-thinking chat, with matching top-five order. Native llama.cpp Q8_0 differs from its own dequantized run by `0.05643513` on the chat's final layer. On the plain prompt, Lisp's second and third logits differ by only `0.00484848` and swap relative to the native oracle.
