
from unsloth_mlx import FastLanguageModel

# ────────────────────────────────────────────────
# 1. Load model & tokenizer
# ────────────────────────────────────────────────
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",  # ← good choice
    max_seq_length = 8192,
    load_in_4bit   = True,
)

# Optional: prepare for faster inference (if the method exists in your version)
try:
    FastLanguageModel.for_inference(model)
except AttributeError:
    pass  # some versions don't have / need this

# ────────────────────────────────────────────────
# 2. Prepare prompt using chat template (recommended for instruct models)
# ────────────────────────────────────────────────
question = """A patient with acute appendicitis has been ill for five days. \
The abdominal pain has slightly lessened but the fever persists, \
and on physical examination a tender mass is felt in the right lower abdomen. \
What should be done at this point?"""

messages = [
    {"role": "user", "content": question}
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# ────────────────────────────────────────────────
# 3. Generate → returns string in recent versions
# ────────────────────────────────────────────────
response = model.generate(
    prompt,                    # ← pass the string directly (not input_ids)
    max_tokens=384,            # increased a bit for medical reasoning
    # temp=0.0,                # many versions ignore temp → greedy by default
    # verbose=True,            # optional – shows progress
)

# ────────────────────────────────────────────────
# 4. Print result
# ────────────────────────────────────────────────
print("\n" + "="*60)
print("Generated response:")
print(response.strip())
print("="*60 + "\n")