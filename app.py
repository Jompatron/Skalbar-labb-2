import os
import gradio as gr
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# ========= CONFIG: EDIT THESE =========
MODEL_REPO_ID = "Jompatron/10k-base-llama"     # HF model repo with the GGUF
MODEL_FILENAME = "merged_model.Q4_K_M.gguf"                 # Exact GGUF file name

N_CTX = 4096   # context window, adjust to match your model / memory
N_THREADS = 4  # tune based on CPU; HF Spaces CPU usually 2–4 cores
N_BATCH = 128  # batch size for prompt processing

# ========= DOWNLOAD MODEL FROM HF HUB =========

def download_model():
    """
    Downloads the GGUF model from Hugging Face Hub (if not already present)
    and returns the local file path.
    """
    local_model_path = hf_hub_download(
        repo_id=MODEL_REPO_ID,
        filename=MODEL_FILENAME,
        local_dir="models",
        local_dir_use_symlinks=False,
    )
    return local_model_path


print("Downloading GGUF model (if needed)...")
model_path = download_model()
print(f"Model downloaded to: {model_path}")

# ========= INITIALIZE LLAMA MODEL =========

print("Loading GGUF model with llama-cpp...")
llm = Llama(
    model_path=model_path,
    n_ctx=N_CTX,
    n_threads=N_THREADS,
    n_batch=N_BATCH,
    # IMPORTANT: set chat_format according to your base model
    # For Llama 3 / Llama 3 Instruct:
    chat_format="llama-3",
)

print("Model loaded successfully.")


# ========= CHAT LOGIC =========

def chat_fn(message, history):
    """
    Gradio ChatInterface callback.

    - message: latest user message (string)
    - history: list of [user_msg, bot_msg] pairs
    """

    # Convert Gradio history -> llama-cpp messages format
    messages = []
    for user_msg, bot_msg in history:
        if user_msg:
            messages.append({"role": "user", "content": user_msg})
        if bot_msg:
            messages.append({"role": "assistant", "content": bot_msg})

    # Add the latest user message
    messages.append({"role": "user", "content": message})

    # Call llama-cpp chat completion
    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=512,     # adjust as needed
        temperature=0.7,
        top_p=0.9,
    )

    reply = response["choices"][0]["message"]["content"]
    return reply


# ========= GRADIO UI =========

chat_ui = gr.ChatInterface(
    fn=chat_fn,
    title="My Fine-tuned Llama 3 (GGUF)",
    description="Chat with my fine-tuned Llama 3 model hosted as a GGUF on Hugging Face.",
    textbox=gr.Textbox(placeholder="Ask me something...", lines=2),
    retry_btn="🔁 Retry",
    undo_btn="↩️ Undo",
    clear_btn="🧹 Clear",
)

if __name__ == "__main__":
    # For local testing
    chat_ui.launch(server_name="0.0.0.0", server_port=7860)
