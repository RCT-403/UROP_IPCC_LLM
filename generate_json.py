# tmp driver to generate json file for testing

import json
from decompo_ipcc import build_report
from finetuning.src.preprocessing.format_data import format_data, create_training_examples

pdf_path = "./data/raw/IPCC AR6 Chapter 2 Climate System.pdf"
report = build_report(pdf_path)
formatted_data = format_data(report.to_dict())
training_examples = create_training_examples(formatted_data)

output_path = "./data/processed/formatted_data.json"
with open(output_path, 'w') as json_file:
    json.dump(formatted_data, json_file, indent=4)

output_path = "./data/processed/training_examples.json"
with open(output_path, 'w') as json_file:
    json.dump(training_examples, json_file, indent=4)