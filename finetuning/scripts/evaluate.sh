#!/bin/bash

# Evaluate the fine-tuned model using the evaluation script

# Activate the virtual environment if needed
# source /path/to/your/venv/bin/activate

# Set the paths for the evaluation script and the processed data
EVALUATE_SCRIPT="../src/evaluation/evaluate.py"
FORMATTED_DATA="../data/processed/formatted_data.json"
TRAIN_VAL_TEST_SPLIT="../data/processed/train_val_test_split.json"

# Run the evaluation script
python $EVALUATE_SCRIPT --formatted_data $FORMATTED_DATA --train_val_test_split $TRAIN_VAL_TEST_SPLIT

echo "Evaluation completed."