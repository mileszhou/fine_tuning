import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset
from difflib import SequenceMatcher
import json

# ----------- CONFIG -------------
base_model_name = "Qwen/Qwen2.5-3B-Instruct"
adapter_path = "./lora_model"
dataset_file = "training_data.json"  # your saved training dataset
max_new_tokens = 200
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
# --------------------------------

# ---- Prompt ----
prompt = "请回答：一个患有急性阑尾炎的病人已经发病5天，腹痛稍有减轻但仍然发热，应如何处理？"


# Load dataset
dataset = load_dataset(
    "FreedomIntelligence/medical-o1-reasoning-SFT",
    "zh",
    split="train[0:1000]",
) # using a small subset for testing — adjust as needed for your experiments
print(f"Loaded {len(dataset)} samples.")

# Load model & tokenizer
tokenizer = AutoTokenizer.from_pretrained(adapter_path)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

# Function to compute similarity
def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

# Iterate over samples
results = []
for idx, sample in enumerate(dataset):
    #prompt = f"### Instruction:\nYou are a medical expert.\n### Question:\n{sample['Question']}\n### Response:\n<think>"
    
    # Tokenize and move to device
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate response
    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=max_new_tokens
    )
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # print("="*50)
    # print(generated)
    # print("-"*30)
    # print(sample["Response"])
    # print("--")
    
    # Compare to original response
    sim = similarity(generated, sample["Response"])
    results.append({
        "index": idx,
        "similarity": sim,
        "generated": generated,
        "original": sample["Response"]
    })
    
    print(f"[{idx}] Similarity: {sim*100:.2f}%")
    if idx >= 20:  # just check first 20 samples for speed
        break

# Optional: save results to analyze later
with open("outputs/memorization_check.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("Done! Results saved to memorization_check.json")
