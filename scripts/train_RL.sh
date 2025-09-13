#!/bin/bash

set -e 

# This script is used to train a large language model (LLM) 
cd "$(dirname "$0")/.." || exit 1

mkdir -p logs/rl

# Execute training script
python -m training.rl.training > logs/rl/training.log
echo "Training completed."
echo "Created by Miguel Angel Cabrera" > logs/rl/training.log
# Execute evaluation script
python -m training.rl.evaluation > logs/rl/evaluations.log
echo "LLM evaluation completed."
echo "Created by Miguel Angel Cabrera" > logs/rl/evaluations.log
