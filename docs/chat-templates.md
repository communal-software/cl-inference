# Chat templates

A chat template turns client messages into the prompt text a model was trained on, and names the tokens that end its reply. Templates are declared with `define-chat-template`, the same inheritance DSL as [architectures](architectures.md) and [tokenizers](tokenizers.md).

```lisp
(local:with-weights (w "Llama-3.2-1B-Instruct-Q8_0.gguf")
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
| `render-chat template messages &key tools add-generation-prompt` | The prompt text; no BOS token |
| `encode-chat template tokenizer messages &rest keys` | Token ids; control tokens become single ids and the generation prompt is added |
| `chat-template-stop-ids template tokenizer` | Ids that end a reply |

`messages` are the [client's](client.md) message plists and `tools` its tool metadata. `render-chat` also takes `:date` (Llama 3) and `:enable-thinking` (Qwen3).

`encode-chat` adds BOS only when the file's `add_bos_token` asks for it, so the prompt never carries two.

## Standard templates

| Template | Detected by | Format | Tools |
|---|---|---|---|
| `llama3-chat` | `<\|start_header_id\|>` and `Cutting Knowledge Date` | Llama 3.2 Instruct | In the first user message; tool results use the `ipython` role; one call per message |
| `chatml` | `<\|im_start\|>` | Plain `<\|im_start\|>role` turns | Rejected |
| `qwen3-chat` | `<\|im_start\|>` and `</think>` | Qwen3 | In the system turn; calls as `<tool_call>` blocks; consecutive results share one `<tool_response>` user turn |

`llama3-chat` and `qwen3-chat` match the official Jinja templates byte for byte.[^jinja]

### Llama 3

Every prompt opens with a system turn holding `Cutting Knowledge Date: December 2023` and `Today Date: ...`. `:date` sets the second line (`"26 Jul 2024"`) and defaults to today. Message text is trimmed.

### Qwen3

An assistant turn keeps its `<think>` block only after the last user query, and always when it is the final message. `:enable-thinking nil` ends the generation prompt with an empty `<think></think>` block.

```lisp
(local:render-chat template messages :add-generation-prompt t :enable-thinking nil)
```

## Selection

`load-chat-template` tries two sources in order:

| Source | Match |
|---|---|
| The file's `tokenizer.chat_template` | Every `:detect` marker of a template occurs in it; the most specific template wins |
| The architecture's `(:chat-template name)` format option | The template class named `name`, in any package |

A file that matches neither signals `unknown-chat-template`.

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
| `(:message control)` | `format` control for one turn, taking the role and the text |
| `(:generation-prompt control)` | Appended when `:add-generation-prompt` is true |
| `(:stop "token" ...)` | Tokens that end a reply |

A child replaces the sections it declares and inherits the rest, as with [tokenizers](tokenizers.md#inheritance). Behaviour that spans messages is a `render-chat` method on the template class:

```lisp
(defmethod local:render-chat ((template my-chat) messages &key &allow-other-keys)
  ...)
```

## Stop ids

`chat-template-stop-ids` returns the ids of the template's `:stop` tokens that the vocabulary holds, then the file's end-of-sequence, end-of-turn and end-of-message ids, without repeats. Llama 3 gives `<|eot_id|>`, `<|eom_id|>` and `<|end_of_text|>`.

## Errors

| Condition | Signalled when |
|---|---|
| `chat-template-error` | Malformed definition, tools given to a template without tool support, a second tool call in one Llama 3 message, or several templates matching one file |
| `unknown-chat-template` | No template matches the file or its architecture; `unknown-chat-template-source` returns the file's text |

## Limitations

| Missing | Tracker |
|---|---|
| Parsing tool calls and reasoning out of a reply | [#36](https://github.com/communal-software/cl-inference/issues/36) |
| Message text that spells a control token encodes as that token | [#37](https://github.com/communal-software/cl-inference/issues/37) |
| Running the file's Jinja directly | not planned |

[^jinja]: The tests compare both against `meta-llama/Llama-3.2-3B-Instruct` and `Qwen/Qwen3-0.6B` as rendered by transformers, tool definitions included. `python-json` writes tool JSON the way Python's `json.dumps(ensure_ascii=False)` does, including float spelling such as `1e-05`. `chatml` omits the default system prompt that older Qwen2 templates add.
