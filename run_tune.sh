#!/usr/bin/env sh

set -e

. .env                      # modify this file to set the default environment variables 
. ./exp_tune/_env/env.sh    # modify this file to set the environment variables specific to fine tuning

run_id=$(date +%Y%m%d_%H%M%S)
result_dir="$RESULTS_DIR/run_$run_id"
echo "Run ID: $run_id"
echo "Results will be saved to: $result_dir"
mkdir -p "$result_dir"

RESULTS_DIR="$result_dir" ./.venv_lora/bin/python ./exp_tune/_python/fine_tuning.py "$@"

# run experiment
#- modify this command to run your experiment, and make sure to save the results to the $result_dir
#./.venv_lora/bin/python ./exp_tune/_python/fine_tuning.py
    # --model_name Qwen/Qwen2.5-3B-Instruct \
    # --dataset_url FreedomIntelligence/medical-o1-reasoning-SFT \
    # --results_dir $result_dir \
    # --model_output_dir _results/fine_tuning/lora_model \
    # --language zh \
    # --num_train_epochs 3 \
    # --per_device_train_batch_size 1 \
    # --data_size 1000
    # --learning_rate 2e-4
