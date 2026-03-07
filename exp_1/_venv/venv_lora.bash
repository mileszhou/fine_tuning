
# Install the venv:
python3 -m venv venv_lora
source venv_lora/bin/activate
pip install --upgrade pip
pip install torch transformers datasets peft trl accelerate

