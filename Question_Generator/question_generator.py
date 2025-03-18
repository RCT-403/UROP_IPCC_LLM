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
file_path = 'summarized_3_3_1_1.txt'
tokenized_content = tokenize_text_file(file_path)

# Convert tokens to a string format if needed
tokenized_string = ' '.join(tokenized_content)

# Prepare the data for the API request
# Use the tokenized content here
# Try to limit the length of the question
# and focus on simple questions/responses
# Can give some sample questions to generate more human-like questions
# Generate Questions and Pair Answers, give some ratings, then we compare the models. 


# data = json.dumps({
#     "model": "deepseek/deepseek-r1:free",
#     "messages": [
#         {
#             "role": "user",
#             "content": f"Based on the following text, generate another insightful question that encourages deeper thinking about the topic with less than {number_of_words} words: {tokenized_string}"
#         }
#     ],
# })

# Send the request
# response = requests.post(
#     url="https://openrouter.ai/api/v1/chat/completions",
#     headers={
#         "Authorization": "Bearer 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",
#         "Content-Type": "application/json",
#     },
#     data=data
# )

# # Print the response
# if response.status_code == 200:
#     response_data = response.json()
#     content = response_data['choices'][0]['message']['content']
#     reason = response_data['choices'][0]['message']['reasoning']
#     print(content)
# else:
#     print(f"Error: {response.status_code} - {response.text}")


for i in range(5):
    for complexity in [3, 5]:
        data = json.dumps({
            "model": "deepseek/deepseek-r1:free",
            "messages": [
                {
                    "role": "user",
                    "content": f"Based on the following text, generate another question with a complexity of {complexity} out of 5 about the topic with less than 15 words: {tokenized_string}"
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
            question_content = response_data['choices'][0]['message']['content']
            print("NEW QUESTION")
            print("Complexity: ", complexity)
            print(question_content)
            
            # write the question content in a new line in questions.txt
            with open('questions.txt', 'a') as file:
                file.write(question_content + '\n')
        
        else:
            print(f"Error: {response.status_code} - {response.text}")
            continue
        
        data = json.dumps({
            "model": "qwen/qwq-32b:free",
            "messages": [
                {
                    "role": "user",
                    "content": f"Given the question {question_content}, generate a response with less than 25 words based on the following text: {tokenized_string}"
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
            answer_content = response_data['choices'][0]['message']['content']
            print("ANSWER TO THE QUESTION")
            print("Complexity: ", complexity)
            print(answer_content)
            
            with open('answers.txt', 'a') as file:
                file.write(answer_content + '\n')
        
        else:
            print(f"Error: {response.status_code} - {response.text}")





