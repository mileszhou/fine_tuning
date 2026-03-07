import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset
from difflib import SequenceMatcher
import json, time, re
from pathlib import Path

# ----------- CONFIG -------------
base_model_name = "Qwen/Qwen2.5-3B-Instruct"
adapter_path = "./_outputs/lora_model"
dataset_file = "training_data.json"  # your saved training dataset
max_new_tokens = 2000
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
dataset_id = "FreedomIntelligence/medical-o1-reasoning-SFT"  # Hugging Face dataset ID
outputs_dir = "./_outputs/compare_training"
results_dir = "./_results/compare_training"
result_fn = f"{results_dir}/comparison.jsonl"
# --------------------------------

def split_sections(text):
    headers = ["Instruction", "Question", "Response"]
    pattern = re.compile(r'(?im)^(?:#+\s*)?(Instruction|Question|Response)\s*:\s*', re.MULTILINE)
    matches = list(pattern.finditer(text))
    sections = {h: "" for h in headers}
    if matches:
        for i, m in enumerate(matches):
            key = m.group(1).capitalize()
            start = m.end()
            end = matches[i+1].start() if i+1 < len(matches) else len(text)
            sections[key] = text[start:end].strip()
    else:
        # fallback: inline labels
        for h in headers:
            m = re.search(rf'{h}\s*:\s*(.*?)(?=\n[A-Z][a-zA-Z_ ]*?:|\Z)', text, re.S)
            if m:
                sections[h] = m.group(1).strip()
    return sections

# Load dataset
dataset = load_dataset(
    dataset_id,
    "zh",
    split="train[0:]",
) 
print(f"• Loaded {len(dataset)} samples.")

# Load model & tokenizer
tokenizer = AutoTokenizer.from_pretrained(adapter_path)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    dtype=torch.float16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, adapter_path)

# base_model.compile()  # compile the base model for faster inference (PyTorch 2.0+)
base_model.eval()
# model.compile()  # compile the model for faster inference (PyTorch 2.0+)
model.eval()
# Move models to the correct device
base_model.to(device)
model.to(device)

# open the outputresult file
r_dir = Path(results_dir)
r_dir.mkdir(parents=True, exist_ok=True)    # ensure results directory exists
f = open(result_fn, "w", encoding="utf-8")

try:
    start = time.perf_counter()
    # Iterate over samples
    for idx, sample in enumerate(dataset):
        prompt = f"### Instruction:\nYou are a medical expert.\n### Question:\n{sample['Question']}\n### Response:\n"
        sample_response = sample['Response']

        # Tokenize and move to device
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        outputs = base_model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens
        )
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

        sections = split_sections(generated)
        instruction = sections["Instruction"]
        question = sections["Question"]
        response = sections["Response"]

        # Generate response
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens
        )
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

        sections = split_sections(generated)
        response_trained = sections["Response"]

        # Save results in JSONL format
        result = {
            "index": idx,
            "question": question,
            "ground_truth": sample_response,
            "response_base": response,
            "response_trained": response_trained,
        }
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
        
        if idx%100==0:
            print(f"Batch @{idx+1} samples in {time.perf_counter() - start:.3f} seconds")
            start = time.perf_counter() 
        
#         f.write(f"# Response from base model ({idx+1}) ==========\n")
#         f.write(f"## Question:\n{Question}\n")
#         f.write(f"### Response from base model:\n{Response}\n")
#         f.write(f"### Response from trained model:\n{Response}\n")
#         f.write(f"### Ground truth response:\n{sample_response}\n")

except KeyboardInterrupt as e:
    print(f"Interrupted at sample {idx+1}")
finally:    # Ensure file is closed even if interrupted
    f.close()
    
print("Done! Results saved to outputs/compare_training.md")
