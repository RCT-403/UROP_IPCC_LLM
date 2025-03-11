from typing import List, Dict
import json

def load_formatted_data(file_path: str) -> Dict[str, str]:
    with open(file_path, 'r') as f:
        return json.load(f)

def evaluate_model(model, validation_data: List[Dict[str, str]]) -> Dict[str, float]:
    # Placeholder for evaluation logic
    results = {}
    for data in validation_data:
        # Simulate model evaluation
        key = data['key']
        paragraph = data['paragraph']
        # Here you would typically call model.evaluate() or similar
        results[key] = len(paragraph)  # Example metric: length of the paragraph
    return results

def main():
    # Load formatted data
    formatted_data_path = './data/processed/formatted_data.json'
    formatted_data = load_formatted_data(formatted_data_path)

    # Convert to list of dicts for evaluation
    validation_data = [{'key': key, 'paragraph': value} for key, value in formatted_data.items()]

    # Load your fine-tuned model here
    # model = load_model('path_to_your_model')

    # Evaluate the model
    evaluation_results = evaluate_model(model=None, validation_data=validation_data)  # Replace None with your model

    # Print evaluation results
    print("Evaluation Results:")
    for key, score in evaluation_results.items():
        print(f"{key}: {score}")

if __name__ == "__main__":
    main()