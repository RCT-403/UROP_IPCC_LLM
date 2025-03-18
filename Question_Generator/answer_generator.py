import requests
import json
import re

# Read the questions.txt file and collect the questions which all start with "**Question:**"
questions = []
with open('questions.txt', 'r') as file:
    for line in file:
        if line.startswith("**Question:**"):
            questions.append(line)

# Read the answers.txt file and collect the answers
answers = []
with open('answers.txt', 'r') as file:
    for line in file:
        answers.append(line)

# Function to read and tokenize text from a file
def tokenize_text_file(file_path):
    with open(file_path, 'r') as file:
        text = file.read()
    tokens = re.findall(r'\b\w+\b', text)  # Matches words
    return tokens

# Path to your text file
file_path = 'summarized_3_3_1_1.txt'
tokenized_content = tokenize_text_file(file_path)

# Convert tokens to a string format if needed
tokenized_string = ' '.join(tokenized_content)

for i in range(10):
    data = json.dumps({
            "model": "deepseek/deepseek-r1:free",
            "messages": [
                {
                    "role": "user",
                    "content": f"Based on the question: {questions[i]} and the response {answers[i]}, give a rating from 1 to 5 on how well the response answers the question based on the text: {tokenized_string}. 1 being the worst and 5 being the best and simply output a number in the content."
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
        rating_content = response_data['choices'][0]['message']['content']
        rating_reason = response_data['choices'][0]['message']['reasoning']
        print(rating_content)
        print(rating_reason)
        
        # write the question content in a new line in questions.txt
        with open('ratings.txt', 'a') as file:
            file.write(rating_content + '\n')

    else:
        print(f"Error: {response.status_code} - {response.text}")