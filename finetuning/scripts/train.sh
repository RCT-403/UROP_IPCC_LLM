#!/bin/bash

# Activate the virtual environment if needed
# source /path/to/your/venv/bin/activate

# Set variables
BASE_MODEL="Deepseek R1 1.5B"
TRAINING_SCRIPT="./src/training/train.py"
CONFIG_DIR="./configs"
LORA_CONFIG="$CONFIG_DIR/lora_config.json"
TRAINING_CONFIG="$CONFIG_DIR/training_config.json"
OUTPUT_DIR="./output"

# Run the training script
python $TRAINING_SCRIPT --lora_config $LORA_CONFIG --training_config $TRAINING_CONFIG --output_dir $OUTPUT_DIR