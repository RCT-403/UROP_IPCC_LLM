# This script contains utility functions for loading and saving formatted data, as well as formatting data to key-paragraph pairs.
from typing import Dict, Any
import json

def load_formatted_data(file_path: str) -> Dict[str, Any]:
    with open(file_path, 'r') as file:
        return json.load(file)

def save_formatted_data(data: Dict[str, Any], file_path: str) -> None:
    with open(file_path, 'w') as file:
        json.dump(data, file)

def format_data_to_key_paragraph(report: Any) -> Dict[str, str]:
    formatted_data = {}
    
    def traverse_section(section: Any):
        if section.content.strip():  # Only include sections with content
            formatted_data[section.title] = section.content.strip()
        for child in section.children:
            traverse_section(child)

    traverse_section(report)
    return formatted_data

def main():
    # Example usage
    report_path = './data/processed/formatted_data.json'
    report = load_formatted_data(report_path)
    formatted_data = format_data_to_key_paragraph(report)
    save_formatted_data(formatted_data, './data/processed/formatted_data.json')

if __name__ == "__main__":
    main()