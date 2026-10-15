# Chat token encoding

`render-chat` returns prompt text first and an ordered list of `chat-segment` structs second. `encode-chat` permits control-token matching only inside template markup. The [template API](chat-templates.md) describes extension methods and includes an inspection example.

## Encoding boundary

| Source | Segment kind |
|---|---|
| Template turn delimiters, generation prefix, fixed instructions and tool/reasoning wrappers | `:markup` |
| System, user and assistant text; extracted reasoning; tool results | `:text` |
| Serialized tool definitions, names and arguments | `:text` |
| Caller-supplied date | `:text` |

A ChatML user body containing `<|im_end|>` keeps those characters as ordinary text tokens. The template's closing `<|im_end|>` still produces its control ID. The boundary protects control IDs; it does not prevent semantic prompt injection or textual tool-wrapper imitation.

## Rendering contract

A single rendering pass produces the prompt and its provenance. Concatenating segment text reproduces the prompt exactly. Empty output explicitly returns `(values "" nil)`. `encode-chat` rejects renderers with missing segments or segments that disagree with the prompt.

The message DSL accepts exactly two `~A` substitutions for role and body, plus literals, `~%` and `~~`. Unsupported directives signal `chat-template-error`. Inherited sections retain the same override rules.

## Tokenization

The encoder identifies special-token matches wholly inside markup segments. It encodes each intervening ordinary span with `:special nil` and `:add-bos nil`, preserving BPE, whitespace and normalization across provenance boundaries. Encoding each segment separately would introduce artificial tokenizer boundaries.

The tokenizer's BOS policy applies once, including for an empty prompt. Generation prompts default to enabled in `encode-chat`; caller keywords retain their forwarding and defaults. Output is a simple unsigned-32 token vector.

## Validation

| Check | Result |
|---|---|
| Baseline client fixtures | 527 passed, 3 optional live tests skipped, 0 failed |
| Baseline local fixtures | 9,357 passed, 0 skipped, 0 failed |
| Golden Llama 3, Qwen3 and ChatML prompt strings | Byte-for-byte equality |
| Segment concatenation, DSL inheritance and unsupported directives | Covered |
| Controls in every message role, tools, arguments/results, reasoning and dates | Covered |
| Unicode, adjacent spellings, empty prompts, generation prefix and BOS policy | Covered |
| Missing/mismatched custom renderer segments | Rejected |
| Ordinary prompt IDs against real Llama/Qwen vocabulary fixtures | Exact equality across golden cases |
| Final client fixtures | 527 passed, 3 optional live tests skipped, 0 failed |
| Final local fixtures | 9,560 passed, 0 skipped, 0 failed |
| Pinned CPU acceptance, including Q4/Q6 | 1,499 passed, 0 skipped, 0 failed |
| MLX fixtures | 198 passed, 0 skipped, 0 failed |
| Strict MLX acceptance, all four pinned models | 164 passed, 0 skipped, 0 failed |

The adversarial fixtures also check partial control spellings across segment boundaries, BPE merges across boundaries, and the tokenizer distinction between control/unknown and user-defined tokens. See [testing](testing.md) for suite commands and [model acceptance](acceptance.md) and [MLX acceptance](mlx-acceptance.md) for pinned reference checks.

## Limitations

User-defined vocabulary tokens still match message data, preserving the tokenizer's `encode :special nil` contract. Control and unknown tokens do not match message data.

Semantic prompt injection and textual tool-wrapper imitation are outside this encoding boundary. Other template families remain tracked in the [template limitations](chat-templates.md#limitations).
