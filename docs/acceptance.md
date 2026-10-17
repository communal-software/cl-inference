# Model acceptance

The acceptance suite compares pinned Llama 3.2 1B, Qwen3 0.6B and [Gemma 3 270M IT](gemma.md#pinned-acceptance) models against llama.cpp. It checks real-model behavior beyond the offline fixtures; see [Testing](testing.md) for routine checks.

[Cache acceptance](cache-acceptance.md) adds exact-prefix reuse and opt-in Q8 KV storage checks against independent Python references.

The shared acceptance helpers use each case’s layer count, template arguments and generation horizon. Qwen3 uses the same numerical bounds as Llama except for the [two Q8_0 cases](#qwen3-reference-checks). All full-model inputs must be present when verifying a model; skips do not establish acceptance.

Q4_0, Q4_K_M and Q6_K variants use [Q4/Q6 acceptance](quant-acceptance.md), with reproducible quantization from the pinned F16 inputs and a F32-dequantized oracle.

## Llama reference checks

`cl-inference/local/acceptance` runs Llama 3.2 1B Instruct in F16 and Q8_0 against llama.cpp: metadata, prompt ids, the residual stream after each layer, the final logits and the greedy ids (at most 8). A plain prompt and a system + user chat run separately. The chat run compares our `llama3-chat` render with llama.cpp rendering the file's own template, and checks that the reply stops on `<|eot_id|>`. The suite also checks Qwen3 0.6B in both formats, with plain prompts and thinking enabled or disabled in chat. It skips every test when its inputs are missing. Cases without their model or dump skip individually.

```sh
curl -L -o reference/models/Llama-3.2-1B-Instruct-f16.gguf \
  https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-f16.gguf
reference/llama-oracle.sh reference/models/Llama-3.2-1B-Instruct-f16.gguf f16 "The capital of France is"
DUMP_SYSTEM="Answer in one word." DUMP_USER="What is the capital of France?" \
  reference/llama-oracle.sh reference/models/Llama-3.2-1B-Instruct-f16.gguf f16-chat ""
curl -L -o reference/models/Llama-3.2-1B-Instruct-Q8_0.gguf \
  https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q8_0.gguf
reference/llama-oracle.sh reference/models/Llama-3.2-1B-Instruct-Q8_0.gguf q8 "The capital of France is"
DUMP_SYSTEM="Answer in one word." DUMP_USER="What is the capital of France?" \
  reference/llama-oracle.sh reference/models/Llama-3.2-1B-Instruct-Q8_0.gguf q8-chat ""
sbcl --dynamic-space-size 16GB --non-interactive --eval '(require :asdf)' --eval '(asdf:test-system :cl-inference/local/acceptance)'
```

The test checks the file against its pinned sha256. `llama-oracle.sh` builds llama.cpp[^oracle] and writes each dump to `reference/oracle/<name>/`; the chat run pins the template's date to 26 Jul 2024, as the test does. llama.cpp multiplies in f16 or Q8_0, ours in f32, so layers match by cosine and relative error, not exactly:

| Model | Cosine at least | Relative error at most | Top logits in order |
|---|---|---|---|
| F16 | 0.9999 | 0.02 | 5 |
| Q8_0 | 0.999 | 0.04 | 3 |

Q8_0 reference comparisons use wider bounds because llama.cpp also quantizes activations.[^llama-drift]

## Qwen3 inputs

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

## Qwen3 reference checks

The acceptance suite compares metadata, exact prompt ids, all 28 layer outputs, final logits, ranked top logits and eight greedy tokens. Each format runs plain completion and chat with thinking enabled and disabled. Inputs missing from `reference/` skip individually; all six Qwen cases must run to establish acceptance.

Qwen3 uses the standard [per-format bounds](#llama-reference-checks) with two case-specific exceptions:

| Q8_0 case | Cosine at least | Relative error at most | Top logits |
|---|---|---|---|
| Plain completion | 0.999 | 0.04 | Exact top token and the same unordered top-three set |
| Non-thinking chat | 0.998 | 0.06 | Top three in order |

F16 and Q8_0 thinking chat retain their standard bounds. Prompt and greedy ids match exactly in every case.[^drift]

Set `DUMP_THINKING=0` for non-thinking chat or `1` for thinking chat when generating reference dumps.[^qwen-oracle]

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

[^llama-drift]: llama.cpp's Q8_0 layers drift 0.026 from its F16 and ours 0.020; in the chat run its fourth and fifth logits sit 0.09 apart and swap.
[^oracle]: CPU-only at release `b11574`, with a small dumper from `reference/dump-trace/` copied into its examples. It records the last position of every `l_out-N` and `result_output` and up to eight greedy ids.
[^qwen-oracle]: The CPU-only llama.cpp `b11574` dumper sets `common_chat_templates_inputs.enable_thinking` from `DUMP_THINKING` before rendering the model's own template. The oracle, its C++ dumper and dumps stay in gitignored `reference/`. They are development tools, never library dependencies. Lisp compares floating-point traces with the per-format bounds documented in [reference bounds](#llama-reference-checks).
[^drift]: llama.cpp's native Q8_0 matmul also quantizes activations. With identical Q8_0 weights dequantized to F32, Lisp's worst layer error is `0.001359204` on the plain prompt and `0.002398626` on non-thinking chat, with matching top-five order. Native llama.cpp Q8_0 differs from its own dequantized run by `0.05643513` on the chat's final layer. On the plain prompt, Lisp's second and third logits differ by only `0.00484848` and swap relative to the native oracle.
