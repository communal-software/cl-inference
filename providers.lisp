(in-package #:cl-inference)

(define-provider :openai
  :protocol :protocol-openai
  :base-url "https://api.openai.com/v1"
  :auth '(:bearer :env "OPENAI_API_KEY")
  :summary "OpenAI chat completions")

(define-provider :ollama
  :protocol :protocol-ollama
  :base-url "http://127.0.0.1:11434"
  :auth :none
  ;; Advertisement, not a gate: the real catalogue is whatever has been
  ;; pulled, which only the running backend knows.
  :models '("llama3.2" "qwen2.5-coder" "gemma3")
  :summary "Local Ollama, native chat endpoint")
