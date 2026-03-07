import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from transformers import TrainingArguments

# --------------------------
# Device
# --------------------------
device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Device:", device)

# --------------------------
# Model selection
# --------------------------
model_name = "Qwen/Qwen2.5-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
) # using float32 for better compatibility with LoRA and to avoid potential issues on MPS with bfloat16 
  # (adjust as needed based on your setup)

model.to(device)    # move model to the correct device (MPS or CPU)

# --------------------------
# Dataset
# --------------------------
dataset = load_dataset(
    "FreedomIntelligence/medical-o1-reasoning-SFT",
    "zh",
    split="train[0:500]",
) # using a small subset for testing — adjust as needed for your experiments

max_seq_length = 1024

def format_example(example):
    text = f"""### Question:
{example["Question"]}

### Response:
{example["Response"]}"""
    return {"text": text}

dataset = dataset.map(format_example)

def tokenize_function(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=max_seq_length,
    )

dataset = dataset.map(tokenize_function, batched=True)
dataset = dataset.remove_columns(
    [col for col in dataset.column_names if col not in ["input_ids", "attention_mask", "labels"]]
)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
#dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])


# --------------------------
# LoRA config
# --------------------------
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
) # these target modules are a good starting point for many models, 
  # but you may want to adjust based on your specific architecture


model = get_peft_model(model, lora_config)  # wrap the model with LoRA adapter

# --------------------------
# Training args
# --------------------------
training_args = TrainingArguments(
    output_dir="./outputs",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    max_steps=100,
    learning_rate=2e-4,
    logging_steps=10,
    optim="adamw_torch",
    report_to="none",
) # these are very basic args for a quick test — adjust as needed for your setup and dataset size

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
)


# --------------------------
# Train
# --------------------------
print("Start training")
trainer.train()
print("End training")

# --------------------------
# Save
# --------------------------
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")
