#!/bin/bash

# This script is used to train a large language model (LLM) 
cd "$(dirname "$0")/.." || exit 

# Execute training script
python training/llm/train_loop.py

echo "Training completed."