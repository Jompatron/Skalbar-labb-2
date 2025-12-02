# Fine-tuned Llama-3 GGUF Chat App

This repository contains a Gradio app that loads a fine-tuned Llama-3 model
stored in GGUF format on Hugging Face and exposes it as a chat interface.

## Files

- `app.py` — main Gradio application using `llama-cpp-python`
- `requirements.txt` — Python dependencies
- (optional) model weights stored separately in a Hugging Face model repo

## How it works

1. On startup, `app.py` downloads the GGUF model file from the Hugging Face Hub
   using `hf_hub_download`.
2. The model is loaded with `llama-cpp-python` using the `chat_format="llama-3"`
   template so prompts are formatted correctly.
3. A `gr.ChatInterface` provides a simple web UI for chatting with the model.

## Running locally

```bash
python -m venv .venv
source .venv/bin/activate      # or .venv\Scripts\activate on Windows

pip install -r requirements.txt

python app.py
