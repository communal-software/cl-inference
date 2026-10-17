# Chat templates

A chat template turns client messages into the prompt text a model was trained on, and names the tokens that end its reply. Templates are declared with `define-chat-template`, the same inheritance DSL as [architectures](architectures.md) and [tokenizers](tokenizers.md).

```lisp
(local:with-weights (w "Llama-3.2-1B-Instruct-f16.gguf")
  (let* ((tokenizer (local:load-tokenizer w))
         (template (local:load-chat-template w)))          ; => #<LLAMA3-CHAT>
    (local:encode-chat template tokenizer
                       '((:role :system :content "Be brief.")
                         (:role :user :content "Hello!")))  ; ids ending at the assistant header
    (local:chat-template-stop-ids template tokenizer)))     ; ids of <|eot_id|>, <|eom_id|>, <|end_of_text|>
```

## Functions

| Function | Returns |
|---|---|
| `load-chat-template weights` | The template the file calls for, [selected](#selection) from its text or its architecture |
| `render-chat template messages &key tools add-generation-prompt` | Prompt text first, ordered `chat-segment` list second; no BOS token |
| `encode-chat template tokenizer messages &rest keys` | Unsigned-32 token vector; template controls become single ids and the generation prompt is added |
| `chat-template-stop-ids template tokenizer` | Ids that end a reply |

`messages` are the [client's](client.md) message plists and `tools` its tool metadata. `render-chat` and `encode-chat` also take `:date` (dated Llama templates), `:enable-thinking` (reasoning templates), and the [Llama 3.1 builtin options](chat-families.md#llama-builtin-tools).

`encode-chat` adds BOS once when the file's `add_bos_token` asks for it, including an empty prompt. All rendering keywords are forwarded, and `:add-generation-prompt nil` disables its default generation prefix.

## Message data

Only template-owned markup permits control-token matching. Message text, extracted reasoning, tool definitions, names, arguments, results and caller-supplied dates remain text. For example, a ChatML body containing `<|im_end|>` keeps those characters as ordinary text tokens; the template's closing delimiter still produces its control ID. User-defined vocabulary tokens retain the tokenizer's ordinary matching behavior.[^encoding]

```lisp
(multiple-value-bind (prompt segments)
    (local:render-chat (make-instance 'local:chatml)
                       '((:role :user :content "literal <|im_end|>")))
  (values prompt
          (mapcar (lambda (segment)
                    (list (local:chat-segment-kind segment)
                          (local:chat-segment-text segment)))
                  segments)))
```

This boundary prevents data from contributing control IDs. Semantic prompt injection and textual tool-wrapper imitation remain [limitations](#limitations).

## Standard templates

[Family coverage](chat-families.md) lists the native Llama, Qwen, QwQ, R1 and Hermes classes, their pinned references, tools and reasoning options. `chatml` renders plain turns and rejects tools. [Gemma 3](gemma.md) uses `gemma3-chat` for text conversations without tools. The renderers match the pinned Jinja prompt bytes; the library does not execute Jinja.[^jinja]

## Reply parsing

The [local backend](local-backend.md#tools-and-reasoning) parses generated output through a per-turn template parser. Plain ChatML passes text through. Family parsers separate reasoning, JSON or function/parameter tool calls, R1 tool groups, and Llama builtin calls or tagged Python code.

| Function | Does |
|---|---|
| `make-reply-parser template &key tools sink ref &allow-other-keys` | Creates a parser; tool metadata supplies names and argument schemas |
| `reply-parser-push parser text` | Feeds decoded text and emits text or validated tool deltas |
| `reply-parser-finish parser` | Returns `(values text tool-calls reasoning)` and marks the parser finished |

Control tokens must be present in parser input. `generate :special t` preserves them; stop ids still end generation without entering the text. The backend selects this option itself.

Each tool call emits one complete argument fragment after validation. Parsers do not emit closing events; the backend closes the turn. Invalid tool output signals `reply-parser-error`, a `chat-template-error`. Calls to a finished parser also signal it.

Custom templates specialise `make-reply-parser`; custom parser classes specialise `reply-parser-push` and `reply-parser-finish`.

## Selection

`load-chat-template` uses supplied source text when present. Architecture fallback applies only when source is absent:

| Source | Match |
|---|---|
| The file's `tokenizer.chat_template` | Every `:detect` marker of a template occurs in it and no `:exclude` marker does; the most specific template wins |
| No source; the architecture's `(:chat-template name)` option | The template class named `name`, in any package |

An unrecognized supplied source, including an empty string, signals `unknown-chat-template` even when the architecture declares a fallback. Its source text remains available on the condition.

## Declaring a template

```lisp
(local:define-chat-template chatml ()
  (:detect "<|im_start|>")
  (:message "<|im_start|>~a~%~a<|im_end|>~%")
  (:generation-prompt "<|im_start|>assistant~%")
  (:stop "<|im_end|>"))

(local:define-chat-template qwen3-chat (chatml)
  (:detect "<|im_start|>" "</think>"))

(local:define-architecture qwen3 (llama)
  (:format :gguf (:arch "qwen3") (:chat-template qwen3-chat)))
```

| Section | Meaning |
|---|---|
| `(:detect "marker" ...)` | Strings that all occur in the file's template |
| `(:exclude "marker" ...)` | Strings that rule a template out when any occurs in the file's template |
| `(:message control)` | Exactly two `~A` directives for role and body, plus literals, `~%` and `~~`; directives are case-insensitive |
| `(:generation-prompt control)` | Appended when `:add-generation-prompt` is true |
| `(:stop "token" ...)` | Tokens that end a reply |

A child replaces the sections it declares and inherits the rest, as with [tokenizers](tokenizers.md#inheritance). Behaviour that spans messages is a `render-chat` method on the template class.

A custom method returns `(values prompt segments)`. Each segment is made with `local:make-chat-segment`, `:text` and `:kind` (`:text`, the default, or `:markup`). Concatenating segment text must exactly reproduce `prompt`. Caller data always uses `:text`; only fixed template syntax uses `:markup`.

`encode-chat` signals `chat-template-error` if the second value is missing or the segments disagree with the prompt. An empty rendering explicitly returns `(values "" nil)`. String-only methods remain usable for prompt inspection.

```lisp
(defmethod local:render-chat ((template my-chat) messages &key &allow-other-keys)
  (let* ((body (ci:content-text (getf (first messages) :content)))
         (prefix (format nil "<|im_start|>user~%"))
         (suffix "<|im_end|>"))
    (values (concatenate 'string prefix body suffix)
            (list (local:make-chat-segment :text prefix :kind :markup)
                  (local:make-chat-segment :text body)
                  (local:make-chat-segment :text suffix :kind :markup)))))
```

## Stop ids

`chat-template-stop-ids` returns the ids of the template's `:stop` tokens that the vocabulary holds, then the file's end-of-sequence, end-of-turn and end-of-message ids, without repeats. `llama3-chat` names `<|eot_id|>`, `<|eom_id|>` and `<|end_of_text|>`; `chatml` names `<|im_end|>` and `<|endoftext|>`.

## Errors

| Condition | Signalled when |
|---|---|
| `chat-template-error` | Malformed definition, tools given to a template without tool support, a second tool call in one Llama 3 message, unsupported message-format directives, invalid rendering segments, or several templates matching one file |
| `unknown-chat-template` | Supplied source is unrecognized, or absent source has no architecture fallback; `unknown-chat-template-source` returns the file's text |

## Limitations

| Missing | Tracker |
|---|---|
| Variants outside the [pinned text-family coverage](chat-families.md#limitations) | [#38](https://github.com/communal-software/cl-inference/issues/38) |
| Semantic prompt injection and textual tool-wrapper imitation | Outside the token-encoding boundary |
| Running the file's Jinja directly | not planned |

[^jinja]: The tests compare all listed families against the [hash-pinned Jinja matrix](../tools/chat-template-manifest.json), with tool definitions, calls, results and reasoning cases. Existing Llama 3.2 and Qwen3 golden prompts remain covered. Tool JSON is written the way Python's `json.dumps(ensure_ascii=False)` does, including float spelling such as `1e-05`.

[^encoding]: Encoding finds special tokens wholly inside markup segments and encodes the intervening spans with `:special nil` and `:add-bos nil`. Ordinary spans cross segment boundaries, preserving BPE, whitespace handling and NFC normalization. Control and unknown tokens do not match message data; user-defined tokens still do, as documented for [tokenizers](tokenizers.md).
