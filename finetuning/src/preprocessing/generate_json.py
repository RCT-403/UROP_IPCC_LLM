'''
- training data generation (formatting)
- train-test split
'''
import json
from finetuning.src.preprocessing.decompose_ipcc import build_report
from finetuning.src.preprocessing.format_data import format_data, create_training_examples
import random

def combine_chapters(chapter_paths, output):
    combined = []
    for path in chapter_paths:
        with open(path, 'r') as f:
            data = json.load(f)
            combined.extend(data)
        
    with open(output, 'w') as f:
        json.dump(combined, f)

def train_test_split(data, train_size, output_train, output_test):
    train = data[:train_size]
    test = data[train_size:]
    
    with open(output_train, 'w') as f:
        json.dump(train, f)
    
    with open(output_test, 'w') as f:
        json.dump(test, f)

if __name__ == '__main__':
    training_examples = []
    for chapter in [2, 3, 4, 11]:
        pdf_path = f"./data/raw/IPCC AR6 Chapter {chapter}.pdf"
        report = build_report(pdf_path)
        
        formatted_data = format_data(report.to_dict())
        output_path = f"./data/processed/formatted_data_{chapter}.json"
        with open(output_path, 'w') as json_file:
            json.dump(formatted_data, json_file, indent=4)

        training_examples.extend(create_training_examples(formatted_data))
    
    random.shuffle(training_examples) # suffle 

    # Save the shuffled training examples
    output_path = "./data/processed/training_examples.json"
    with open(output_path, 'w') as json_file:
        json.dump(training_examples, json_file, indent=4)

    # Optionally perform train-test split
    train_size = int(0.8 * len(training_examples))
    train_test_split(training_examples, train_size, 
                    "./data/processed/train.json",
                    "./data/processed/test.json")