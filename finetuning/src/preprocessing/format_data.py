from typing import Dict, Any, List
import json
import re

def format_data(report_data: Dict[str, Any]) -> Dict[str, str]:
    """
    Format the parsed IPCC report data into a dictionary with meaningful keys.
    Keys will be formatted as "section_number - section_title" to preserve both
    indexing structure and semantic meaning.
    """
    formatted_data = {}
    
    def process_section(section: Dict[str, Any]) -> None:
        # Create a meaningful key combining section number and title
        index = section.get("index")
        title = section.get("title")
        
        if index and title:
            # Extract just the title without the section number if it's included in the title
            title_only = re.sub(r'^\s*\d+(\.\d+)*\s*', '', title).strip()
            key = f"{index} - {title_only}"
            
            # Only include sections with actual content
            content = section.get("content", "").strip()
            if content:
                formatted_data[key] = content
        
        # Process children recursively
        for child in section.get("children", []):
            process_section(child)
    
    # Start processing from the root
    process_section(report_data)
    
    return formatted_data

def create_training_examples(formatted_data: Dict[str, str]) -> List[Dict[str, str]]:
    """
    Convert the formatted data into a list of training examples suitable for fine-tuning.
    Each example contains a prompt (the section title) and completion (the content).
    """
    training_examples = []
    
    for key, content in formatted_data.items():
        if not content:
            continue
            
        example = {
            "prompt": f"Summarize information from IPCC report section: {key}\n\n",
            "completion": content
        }
        training_examples.append(example)
        
    return training_examples

def save_formatted_data(data: Dict[str, str], output_path: str) -> None:
    with open(output_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)
        
def save_training_examples(examples: List[Dict[str, str]], output_path: str) -> None:
    with open(output_path, 'w') as json_file:
        json.dump(examples, json_file, indent=4)

def main(parsed_data_path: str, formatted_output_path: str, training_examples_path: str) -> None:
    with open(parsed_data_path, 'r') as file:
        parsed_data = json.load(file)
    
    formatted_data = format_data(parsed_data)
    save_formatted_data(formatted_data, formatted_output_path)
    
    training_examples = create_training_examples(formatted_data)
    save_training_examples(training_examples, training_examples_path)
    
    print(f"Processed {len(formatted_data)} sections")
    print(f"Created {len(training_examples)} training examples")

if __name__ == "__main__":
    parsed_data_path = "../data/processed/parsed_data.json"
    formatted_output_path = "../data/processed/formatted_data.json"
    training_examples_path = "../data/processed/training_examples.json"
    main(parsed_data_path, formatted_output_path, training_examples_path)