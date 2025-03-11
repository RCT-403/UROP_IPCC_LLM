#!/bin/bash

# Preprocessing script for IPCC report fine-tuning

# Step 1: Parse the PDF and extract content
python src/preprocessing/format_data.py --input data/raw/IPCC_AR6_WGI_Chapter_2.pdf --output data/processed/formatted_data.json

# Step 2: Split the formatted data into training, validation, and test sets
python src/preprocessing/data_utils.py --input data/processed/formatted_data.json --output data/processed/train_val_test_split.json

echo "Preprocessing completed. Formatted data and splits are saved."