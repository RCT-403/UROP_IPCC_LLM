import requests
import json
import re

# Function to read and tokenize text from a file
def tokenize_text_file(file_path):
    with open(file_path, 'r') as file:
        text = file.read()
    tokens = re.findall(r'\b\w+\b', text)  # Matches words
    return tokens

# Path to your text file
file_path = 'your_file.txt'
tokenized_content = tokenize_text_file(file_path)

# Convert tokens to a string format if needed
tokenized_string = ' '.join(tokenized_content)

# Prepare the data for the API request
data = json.dumps({
    "model": "cognitivecomputations/dolphin3.0-mistral-24b:free",
    "messages": [
        {
            "role": "user",
            "content": f"Based on the following text, generate an insightful question that encourages deeper thinking about the topic: {tokenized_string}"  # Use the tokenized content here
        }
    ],
})

# Send the request
response = requests.post(
    url="https://openrouter.ai/api/v1/chat/completions",
    headers={
        "Authorization": "Bearer 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",
        "Content-Type": "application/json",
    },
    data=data
)

# Print the response
if response.status_code == 200:
    response_data = response.json()
    content = response_data['choices'][0]['message']['content']
    print(content)
else:
    print(f"Error: {response.status_code} - {response.text}")
