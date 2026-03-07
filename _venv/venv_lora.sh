#!/usr/bin/env sh
set -e

python3 -m venv .venv_lora

PATH=.venv_lora/bin:$PATH

pip install --upgrade pip
pip install torch transformers datasets peft trl accelerate
pip install ipykernel
pip install openai
