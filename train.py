import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer, SFTConfig


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DATASET_NAME = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
OUTPUT_DIR = "/content/drive/MyDrive/Qwen2.5-1.5B"
MAX_SEQ_LENGTH = 512                    # [if you have more VRAM] raise to 1024/2048

# Optional: cap dataset size for faster iteration on a 4GB card.
# Full dataset is ~26.9K rows; 3 epochs over all of it will take a long time
# on a 4GB GPU. Set to None to use the full dataset.
MAX_TRAIN_EXAMPLES = 5000                # [if you have time/more VRAM] set to None

# Optional: train on a subset of categories only (e.g. just ORDER-related intents).
# Set to None to use all 10 categories.
# Options: ACCOUNT, CANCELLATION_FEE, DELIVERY, FEEDBACK, INVOICE, NEWSLETTER,
#          ORDER, PAYMENT, REFUND, SHIPPING_ADDRESS
CATEGORY_FILTER = None


# Quantization config — this is what makes a 3B model fit in 4GB.
# NF4 is the quantization scheme from the QLoRA paper; double_quant saves
# a bit more memory by quantizing the quantization constants themselves.

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,   # compute dtype for matmuls
    bnb_4bit_use_double_quant=True,
)


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",            # places the model on your single GPU
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)


# Required prep step for k-bit (4-bit/8-bit) training: casts norm layers to
# fp32 for stability and enables gradient checkpointing-friendly inputs.

model = prepare_model_for_kbit_training(model)
model.config.use_cache = False  # must be False during training w/ gradient checkpointing
model.gradient_checkpointing_enable()  # trades compute for memory — essential at 4GB


lora_config = LoraConfig(
    r=16,                       # [if you have more VRAM] try 32 or 64 for more capacity
    lora_alpha=32,              # commonly set to 2x r
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# Load the Bitext dataset and convert to chat format.
# Raw columns: flags, instruction, category, intent, response.
# We build a system prompt + user/assistant turn per example, then apply
# the model's chat template — never hand-build prompt strings yourself.
# Qwen2.5 uses ChatML-style special tokens internally; apply_chat_template
# handles that automatically and correctly.

SYSTEM_PROMPT = "You are a helpful customer support assistant."

raw_dataset = load_dataset(DATASET_NAME, split="train")

if CATEGORY_FILTER is not None:
    raw_dataset = raw_dataset.filter(lambda ex: ex["category"] == CATEGORY_FILTER)

if MAX_TRAIN_EXAMPLES is not None:
    raw_dataset = raw_dataset.shuffle(seed=42).select(
        range(min(MAX_TRAIN_EXAMPLES, len(raw_dataset)))
    )

print(f"Using {len(raw_dataset)} examples from {DATASET_NAME}")


def to_chat_format(example):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["response"]},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}

dataset = raw_dataset.map(to_chat_format, remove_columns=raw_dataset.column_names)

# Hold out a small validation split to watch for overfitting
split = dataset.train_test_split(test_size=0.05, seed=42)
train_dataset = split["train"]
eval_dataset = split["test"]


# Training config
# batch_size=1 + gradient_accumulation is the standard way to simulate a
# larger effective batch size on tiny VRAM budgets.

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=2,                     # 2 epochs is plenty on 5000 diverse examples; # 3+ risks overfitting/memorizing responses
    per_device_train_batch_size=1,          # do not raise this on 4GB
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,          # effective batch size = 1 * 8 = 8
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",               # paged optimizer avoids OOM spikes
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,
    bf16=True,                              # [if no bf16 support] set fp16=True instead
    max_length=MAX_SEQ_LENGTH,
    dataset_text_field="text",
    packing=False,                          # keep False for clearer per-example loss
    report_to="none",                       # set to "wandb" if you use Weights & Biases
)


trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
)


import os

# Auto-detect: resume if a checkpoint already exists in OUTPUT_DIR (e.g. you're
# re-running this cell after a Colab disconnect), otherwise start fresh.
# (resume_from_checkpoint=True errors out if no checkpoint exists yet, so we
# check for one first rather than hardcoding True/False.)
has_checkpoint = os.path.isdir(OUTPUT_DIR) and any(
    name.startswith("checkpoint-") for name in os.listdir(OUTPUT_DIR)
)

if has_checkpoint:
    print(f"Found existing checkpoint in {OUTPUT_DIR} — resuming training.")
else:
    print(f"No checkpoint found in {OUTPUT_DIR} — starting fresh training run.")

trainer.train(resume_from_checkpoint=has_checkpoint)

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\nDone. LoRA adapter saved to: {OUTPUT_DIR}")
