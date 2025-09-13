#!/bin/bash

set -e 

# This script is used to train a large language model (LLM) 
cd "$(dirname "$0")/.." || exit 1

mkdir -p images/llm
mkdir -p images/rl
mkdir -p checkpoints
mkdir -p logs/llm
# Execute training script
python -m training.llm.train_loop > logs/llm/training.log
echo "Training completed."
echo "Created by Miguel Angel Cabrera" > logs/llm/training.log

python -m training.llm.evaluation > logs/llm/evaluations.log
echo "LLM evaluation completed."
echo "Created by Miguel Angel Cabrera" > logs/llm/evaluations.log
