from transformers import AutoTokenizer
from peft import PeftModel
from transformers import AutoModelForCausalLM
import torch

base_model_name = "Qwen/Qwen2.5-3B-Instruct"
adapter_path = "./lora_model"

# ---- Prompt ----
prompt = "请回答：一个患有急性阑尾炎的病人已经发病5天，腹痛稍有减轻但仍然发热，应如何处理？"


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(adapter_path)

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,  # or "auto"
    device_map="auto"
)

# Tokenize
inputs = tokenizer(prompt, return_tensors="pt").to(base_model.device)

# Generate
outputs = base_model.generate(
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    max_new_tokens=200,
    temperature=0.7,
)

# Decode
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("# Pretrained model")
print(response)
print("-"*20)

# Load LoRA adapter
#
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()


# Tokenize
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate
outputs = model.generate(
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    max_new_tokens=200,
    temperature=0.7,
)

# Decode
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("# Trained model")
print(response)
print("-"*20)
