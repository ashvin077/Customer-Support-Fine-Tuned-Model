"""
Customer Support Chatbot — Streamlit App
Loads a fine-tuned (QLoRA/PEFT) Qwen2.5-1.5B-Instruct model and serves a chat UI.

Folder structure expected:
    project/
    ├── app.py
    └── model/                  <- your fine-tuned adapter + tokenizer files
        ├── adapter_config.json
        ├── adapter_model.safetensors
        ├── tokenizer.json
        ├── tokenizer_config.json
        └── ...

If you instead saved a FULLY MERGED model (no adapter_config.json present),
set USE_PEFT_ADAPTER = False below.
"""

import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ──────────────────────────────────────────────────────────────────────────
# CONFIG — edit these to match your project
# ──────────────────────────────────────────────────────────────────────────
BASE_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"   # base model the adapter was trained on
ADAPTER_PATH = "Qwen2.5-1.5B/"                          # local folder with your fine-tuned adapter/tokenizer
USE_PEFT_ADAPTER = True                         # False if you saved a merged full model instead
LOAD_IN_4BIT = True                             # keep True if running on limited VRAM (e.g. T4/3050)
MAX_NEW_TOKENS = 256
SYSTEM_PROMPT = (
    "You are a helpful, professional customer support assistant. "
    "Answer clearly and concisely, and ask clarifying questions when needed."
)

st.set_page_config(page_title="Customer Support Chatbot", page_icon="💬", layout="centered")


# ──────────────────────────────────────────────────────────────────────────
# MODEL LOADING (cached so it only loads once per session/server)
# ──────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    quant_config = None
    if LOAD_IN_4BIT and device == "cuda":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    # Tokenizer: load from the fine-tuned folder so any added/special tokens are preserved
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)

    if USE_PEFT_ADAPTER:
        from peft import PeftModel

        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            quantization_config=quant_config,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        )
        model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            ADAPTER_PATH,
            quantization_config=quant_config,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        )

    if device == "cpu":
        model.to(device)

    model.eval()
    return model, tokenizer, device


def generate_response(model, tokenizer, device, chat_history):
    """chat_history: list of {"role": "user"/"assistant", "content": str}"""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + chat_history

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return response.strip()


# ──────────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────────
st.title("💬 Customer Support Assistant")
st.caption("Fine-tuned Qwen2.5-1.5B-Instruct · QLoRA")

with st.spinner("Loading model… this can take a minute on first run."):
    model, tokenizer, device = load_model_and_tokenizer()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render existing chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Sidebar controls
with st.sidebar:
    st.subheader("Settings")
    st.write(f"**Device:** {device.upper()}")
    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.rerun()

# Chat input
user_input = st.chat_input("Type your question…")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            reply = generate_response(model, tokenizer, device, st.session_state.messages)
        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
