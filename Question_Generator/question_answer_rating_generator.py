import requests
import json
import re

key = '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'

# Function to read and tokenize text from a file
def tokenize_text_file(file_path):
    with open(file_path, 'r') as file:
        text = file.read()
    tokens = re.findall(r'\b\w+\b', text)  # Matches words
    return tokens

# Tokenize the text from the file
file_path = 'summarized_3_3_1_1.txt'
tokenized_content = tokenize_text_file(file_path)
tokenized_string = ' '.join(tokenized_content)

def generate_question(tokenized_string, complexity):
    data = json.dumps({
        "model": "deepseek/deepseek-r1:free",
        "messages": [
            {
                "role": "user",
                "content": f"""Please generate 5 questions from the following text with a complexity level of {complexity} out of 5:
{tokenized_string} \n Please follow the steps below to process the climate report snippet:
    1. read and understand the report carefully
    2. identify the key weather/climate related information in the report
    3. generate a high quality question based on the key information
    4. use logical reasoning to arrive at the correct answer
    5. The final output must strictly use the following JSON format :("Question": "You asked a question", "answer": "answer you gave","logic" : "The logic of your reasoning"),
    Make sure to use double quotes for all keys and string values
    Note: Only output the final JSON"""
            }
        ],
    })

    # Send the request
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        data=data
    )

    # Print the response
    if response.status_code == 200:
        response_data = response.json()
        content = response_data['choices'][0]['message']['content']
        reason = response_data['choices'][0]['message']['reasoning']
        return content
    else:
        return f"Error: {response.status_code} - {response.text}"

print("test")
print(generate_question(tokenized_string, 3))

