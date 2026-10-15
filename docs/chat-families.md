# Chat template families

The native Lisp renderers cover the pinned text templates below. Loading a template does not establish support for that model's architecture, tokenizer or modalities. See [model support](support.md) and [template selection](chat-templates.md#selection).

## Coverage

| Class | Reference | Tools and reasoning |
|---|---|---|
| `chatml` | Plain ChatML turns | Text only |
| `llama30-chat` | Llama 3.0 8B Instruct | Trimmed turns, no date header; tools rejected |
| `llama31-chat` | Llama 3.1 8B Instruct | Date header, function JSON, explicit builtin mode; one call per message |
| `llama3-chat` | Llama 3.2 3B Instruct | Date header, function JSON and tagged Python |
| `qwen25-chat` | Qwen2.5 7B Instruct | Default Qwen system prompt, JSON tool calls and grouped results |
| `qwen3-chat` | Qwen3 0.6B | JSON tool calls and reasoning history |
| `qwq-chat` | QwQ 32B | JSON tool calls; generation prefix opens reasoning |
| `r1-chat` | R1 Distill Qwen 32B | DeepSeek control tokens, tool groups and reasoning |
| `r1-llama-chat` | R1 Distill Llama 8B | Distinct R1 tool-history rendering; generation prefix opens reasoning |
| `qwen3-coder-chat` | Qwen3-Coder | XML-like tool schemas and function/parameter calls |
| `qwen35-chat` | Qwen3.5 4B | Text-only content, function/parameter calls and reasoning |
| `hermes-chat` | Hermes 2 Pro Llama 3 and Hermes 3 Llama 3.1 tool-use | Shared reference format, Python-style schema descriptions and JSON calls |

Nonempty message lists are required by the Qwen and dated Llama templates. Qwen3.5 also requires a user query and permits a system message only at the beginning. ChatML, Llama 3.0, Hermes and R1 accept empty histories.

Prompt bytes, source detection, generation prefixes and errors are checked against [hash-pinned references](../tools/chat-template-manifest.json). BOS belongs to the tokenizer; renderers omit it. The [fixture generator](../tools/chat-template-reference.py) uses development-only Python dependencies.[^references]

R1 tool definitions are not injected into the prompt by these reference templates; include any required tool instructions in the system message.

## Llama builtin tools

`:builtin-tools` is a list of distinct keyword names present in `:tools`. Supplying it selects Llama 3.1 builtin mode, including when the list is empty. `:tools-in-user-message` defaults to true; false places function definitions in the system turn.

```lisp
(local:render-chat
 (make-instance 'local:llama31-chat)
 '((:role :user :content "Find information about Lisp."))
 :tools '((:name :brave_search :summary "Search"
           :params ((:query string :required t))))
 :builtin-tools '(:brave_search)
 :tools-in-user-message nil
 :date "26 Jul 2024"
 :add-generation-prompt t)
```

The local client accepts the same options. Builtin calls such as `<|python_tag|>brave_search.call(query="Lisp")` become ordinary validated tool events. The parser reads the call syntax without evaluating code. Tool execution remains the caller's responsibility.

Tagged Python uses the existing `:python` tool with a `:code` string. In builtin mode, a declared `:code_interpreter` tool with a `:code` string receives code instead.

## Reasoning

`:enable-thinking` defaults to true. QwQ, Qwen3.5 and R1 generation prefixes open `<think>`; their parsers start inside reasoning. Pass the same thinking option to rendering and `make-reply-parser`. The local backend forwards it to both.

An unfinished thinking block remains reasoning at a generation limit. `:enable-thinking nil` closes the generated thinking prefix for QwQ, Qwen3.5 and R1 Distill Qwen. The pinned R1 Distill Llama template always opens reasoning. Existing Qwen3 behavior uses its own generation prefix and history rules.

```lisp
(local:render-chat (make-instance 'local:qwen25-chat)
                   '((:role :user :content "Hello!"))
                   :add-generation-prompt t)
```

This includes Qwen2.5's default system text. An explicit system message replaces that default.

## Limitations

- Coverage is for the pinned text variants. Other source variants and multimodal inputs require separate validation: [template coverage](https://github.com/communal-software/cl-inference/issues/38).
- Detection uses marker strings; it does not prove arbitrary Jinja programs equivalent.
- Builtin call history requires string arguments without quotes, backslashes or line breaks, matching the pinned template's unescaped format. Generated builtin calls accept JSON-escaped strings.
- Function/parameter wrappers do not escape their delimiters. Literal closing wrappers in argument strings are ambiguous; schema types determine scalar decoding, including `None` for nullable values. An `any` parameter uses JSON when valid and literal text otherwise. Numeric-looking strings and nullable literal `None` can lose their original type: [explicit argument codecs](https://github.com/communal-software/cl-inference/issues/56).
- The pinned Hermes template rejects nullable schema types. Reference quirks are retained: Hermes describes array parameters as `list[Union[]]`; R1 Distill Llama omits the closing tool-group marker after the first call in history. These are prompt-reference behaviors, not evidence of model execution support.
- Template controls cannot be introduced through caller text; semantic prompt injection and textual wrapper imitation remain outside that boundary.

[^references]: The manifest records immutable source URLs and SHA-256 hashes for every vendored test template. llama.cpp supplies the family matrix; Llama 3.0 comes from the pinned NousResearch tokenizer configuration. Jinja2 3.1.6 and MarkupSafe 3.0.4 are pinned in the development requirements. Expected prompt fixtures run offline and contain no runtime Python dependency.
