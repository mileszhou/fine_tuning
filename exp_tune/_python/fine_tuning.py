import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from transformers import TrainingArguments
import time, os
from pathlib import Path
import argparse
import json
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model

# Arguments:
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default=os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-3B-Instruct"))
parser.add_argument("--dataset_url", default=os.environ.get("DATA_SETURL", "FreedomIntelligence/medical-o1-reasoning-SFT"))
parser.add_argument("--language", default=os.environ.get("LANGUAGE", "zh"))
parser.add_argument("--results_dir", default=os.environ.get("RESULTS_DIR", "./_results/fine_tuning/run_vscode"))
parser.add_argument("--num_train_epochs", type=int, default=os.environ.get("NUM_TRAIN_EPOCHS", 3))
parser.add_argument("--per_device_train_batch_size", type=int, default=os.environ.get("PER_DEVICE_TRAIN_BATCH_SIZE", 4))
parser.add_argument("--gradient_accumulation_steps", type=int, default=os.environ.get("GRADIENT_ACCUMULATION_STEPS", 2))
parser.add_argument("--max_seq_length", type=int, default=os.environ.get("MAX_SEQ_LENGTH", 2048))
parser.add_argument("--lora_rank", type=int, default=os.environ.get("LORA_RANK", 16))
parser.add_argument("--lora_alpha", type=int, default=os.environ.get("LORA_ALPHA", 32))
parser.add_argument("--lora_dropout", type=float, default=os.environ.get("LORA_DROPOUT", 0.05))
parser.add_argument("--learning_rate", type=float, default=os.environ.get("LEARNING_RATE", 2e-4))
parser.add_argument("--data_size", type=int, default=os.environ.get("DATA_SIZE", 1000))  # number of samples to use from the dataset (for quick testing)

args = parser.parse_args()
json_args = vars(args)
print(json.dumps(json_args, indent=4, ensure_ascii=False))

model_name = args.model_name
dataset_url = args.dataset_url
language = args.language
num_train_epochs = args.num_train_epochs
per_device_train_batch_size = args.per_device_train_batch_size
learning_rate = args.learning_rate
data_size = args.data_size
lora_rank = args.lora_rank
lora_alpha = args.lora_alpha
lora_dropout = args.lora_dropout
results_dir = args.results_dir
model_output_dir = os.path.join(args.results_dir, "lora_model")

result_path = Path(results_dir)
result_path.mkdir(parents=True, exist_ok=True)    # ensure results directory exists
with open(os.path.join(args.results_dir, "training_parameters.json"), "w", encoding="utf-8") as f:
    json.dump(json_args, f, indent=4, ensure_ascii=False)
    f.flush()
    f.close()

# exit(0)  # for testing argument parsing and logging

# --------------------------
# Device
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

tokenizer = AutoTokenizer.from_pretrained(model_name)

# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     dtype=torch.bfloat16,
# ) # using float32 for better compatibility with LoRA and to avoid potential issues on MPS with bfloat16 
#   # (adjust as needed based on your setup)

# --------------------------
# Dataset
# --------------------------
dataset = load_dataset(
    "FreedomIntelligence/medical-o1-reasoning-SFT",
    "zh",
    split=f"train[0:{data_size}]"
) # using a small subset for testing — adjust as needed for your experiments
print(f"Loaded {len(dataset)} samples.")

max_seq_length = 2048

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
    r=lora_rank,
    lora_alpha=lora_alpha,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
) # these target modules are a good starting point for many models, 
  # but you may want to adjust based on your specific architecture

from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="bfloat16",
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)


model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    r=lora_rank,
    lora_alpha=lora_alpha,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=lora_dropout,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
# model.to(device)    # move model to the correct device (MPS or CPU)

# model = get_peft_model(model, lora_config)  # wrap the model with LoRA adapter

# --------------------------
# Training args
# --------------------------
training_args = TrainingArguments(
# 
#     num_train_epochs=1,                     # Start here – monitor loss
#     per_device_train_batch_size=16,         # ↑ this with your memory headroom
#     gradient_accumulation_steps=1,          # No need to accumulate now
#     learning_rate=1e-4 or 2e-5,             # Common starting points for LoRA on Qwen2.5
#     lr_scheduler_type="cosine",             # Smooth decay
#     warmup_ratio=0.03,                      # ~3% warmup
#     weight_decay=0.01,                      # Light regularization
#     fp16=True or bf16=True,                 # If using torch (Apple Silicon supports bf16 well)
#     logging_steps=50,                       # Frequent logs
#     save_strategy="steps",
#     load_best_model_at_end=True,            # If using val loss
#     # ... other args like max_grad_norm=1.0 for stability

    output_dir="./outputs.tmp",
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=2,          # Accumulate gradients over 2 steps to effectively double batch size
    num_train_epochs=num_train_epochs,
    learning_rate=learning_rate,
    logging_steps=100,
    lr_scheduler_type="cosine",             # Smooth decay
    warmup_ratio=0.03,                      # ~3% warmup
    weight_decay=0.01,                      # Light regularization
    bf16=True,                 # If using torch (Apple Silicon supports bf16 well)
    #max_steps= 30000,
    #evaluation_strategy="steps",            # Or "epoch" if you have val set
    save_strategy="steps",     # important
    save_steps=100,            # save every 1000 steps
    save_total_limit=3,        # keep last 3 checkpoints
    report_to="none",
)
# these are very basic args for a quick test — adjust as needed for your setup and dataset size
# trainer.train(resume_from_checkpoint=True)    # if resuming is needed.
# trainer.train(resume_from_checkpoint="./outputs/checkpoint-3000")


trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
)

# Check trainer
print(f"Trainer's data length: {trainer.get_train_dataloader().__len__()}.")
# number_of_steps_per_epoch =
#     len(train_dataset) / per_device_train_batch_size / gradient_accumulation_steps


# --------------------------
# Train
# --------------------------
print("Start training")
start = time.perf_counter()
try:
    trainer.train()
except KeyboardInterrupt:
    print("Interrupted! Saving checkpoint...")

end = time.perf_counter()
elapsed = end - start
print(f"Elapsed time: {elapsed:.3f} seconds")

# --------------------------
# Save
# --------------------------
trainer.save_model(f"{model_output_dir}")
tokenizer.save_pretrained(f"{model_output_dir}")

# Cleanup to avoid a segment fault on MPS 
del trainer
del model
torch.cuda.empty_cache()
