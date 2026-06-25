import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import mlflow


# for inference (generation)

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_PATH = ".\Qwen2.5-1.5B"
SYSTEM_PROMPT = "You are a helpful customer support assistant."

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

print("Chatbot ready. Type 'exit' to quit.\n")

history = [{"role": "system", "content": SYSTEM_PROMPT}]
while True:
    user_input = input("You: ").strip()
    if user_input.lower() == "exit":
        break

    history.append({"role": "user", "content": user_input})

    prompt = tokenizer.apply_chat_template(
        history, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    print(f"\nUser Input: {user_input}\nBot: {response}\n")
    history.append({"role": "assistant", "content": response})






# DEPLOY AND RESISTER MODEL IN MLFLOW

training_params = {
    "num_train_epochs" : 2,                     # 2 epochs is plenty on 5000 diverse examples; # 3+ risks overfitting/memorizing responses
    "per_device_train_batch_size" : 1,          # do not raise this on 4GB
    "per_device_eval_batch_size" : 1,
    "gradient_accumulation_steps" : 8,          # effective batch size = 1 * 8 = 8
    "gradient_checkpointing" : "True",
    "optim" : "paged_adamw_8bit",               # paged optimizer avoids OOM spikes
    "learning_rate" : "2e-4",
    "lr_scheduler_type" : "cosine",
    "warmup_ratio" : 0.03,
    "logging_steps" : 10,
    "eval_strategy" : "steps",
    "eval_steps" : 100,
    "save_strategy" : "steps",
    "save_steps" : 100,
    "save_total_limit" : 3,
    "bf16":"True",                              # [if no bf16 support] set fp16=True instead
    "max_length" : 512,
    "dataset_text_field" : "text",
    "packing" : "False",                          # keep False for clearer per-example loss
    "report_to" : "none"
}


metrics = {
    "training loss" : 0.548204, 
    "evaluation loss" : 0.617285,
    "Entropy" : 0.562029,
    "Accuracy" : 0.810830
}


# Merge adapter into base model
merged_model = model.merge_and_unload()

# Save merged model
merged_model.save_pretrained("./merged_model")
tokenizer.save_pretrained("./merged_model")


mlflow.set_experiment("Customer-Support-Chat-Bot")
mlflow.set_tracking_uri("http://127.0.0.1:5000/")

with mlflow.start_run(run_name="first-cs-chat-model"):
    mlflow.log_params(training_params)
    
    mlflow.log_metrics({
        "Training Loss" : metrics["training loss"],
        "Evaluation Loss" : metrics["evaluation loss"],
        "Entropy" : metrics["Entropy"],
        "Accuracy" : metrics["Accuracy"]
    })
    
    mlflow.transformers.log_model(
        transformers_model={
            "model" : merged_model,
            "tokenizer" : tokenizer
        },
        name = "Customer-Support-ChatBot-Qwen2_5-1_5B-fine-tuned-model"
    )

# model registry
model_name = "Customer-Support-ChatBot-Qwen2_5-1_5B-fine-tuned-model"
run_id = "93c18152a5bd47f6a37f3ca3df41b1e2"
model_uri = f"runs:/{run_id}/{model_name}"
result = mlflow.register_model(
    model_uri, model_name
)



# DEPLOY AND REGISTER MODEL IN DAGSHUB

import dagshub
dagshub.init(repo_owner="iam.ashvindhakal", repo_name="Customer-Support-ChatBot" , mlflow=True)


mlflow.set_experiment("Customer-Support-Chat-Bot")
mlflow.set_tracking_uri("https://dagshub.com/iam.ashvindhakal/Customer-Support-ChatBot.mlflow")

with mlflow.start_run(run_name="first-cs-chat-model"):
    mlflow.log_params(training_params)
    
    mlflow.log_metrics({
        "Training Loss" : metrics["training loss"],
        "Evaluation Loss" : metrics["evaluation loss"],
        "Entropy" : metrics["Entropy"],
        "Accuracy" : metrics["Accuracy"]
    })
    
    mlflow.transformers.log_model(
        transformers_model={
            "model" : merged_model,
            "tokenizer" : tokenizer
        },
        name = "Customer-Support-ChatBot-Qwen2_5-1_5B-fine-tuned-model"
    )
