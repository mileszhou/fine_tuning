#!/usr/bin/env sh
set -e

python3 -m venv .venv_lora
. .venv_lora/bin/activate

PATH=.venv_lora/bin:$PATH

pip install --upgrade pip
pip install torch transformers datasets peft trl accelerate
pip install ipykernel
pip install openai
pip install bitsandbytes

source _env                      # modify this file to set the default environment variables