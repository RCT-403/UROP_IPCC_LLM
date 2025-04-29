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

def generate_question(tokenized_string, complexity, number_of_questions=5):
    data = json.dumps({
        "model": "qwen/qwq-32b:free",
        "messages": [
            {
                "role": "user",
                "content": f"""Please generate {number_of_questions} questions from the following text with a complexity level of {complexity} out of 5:
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
        return json.loads(content)
    else:
        return f"Error: {response.status_code} - {response.text}"

def generate_answer(tokenized_string, question):
    data = json.dumps({
        "model": "deepseek/deepseek-r1:free",
        "messages": [
            {
                "role": "user",
                "content": f"""Based on the question: {question}, generate an answer based on the text: {tokenized_string}. 
                Please follow the steps below to process the climate report snippet:
    1. read and understand the report carefully
    2. identify the key weather/climate related information in the report
    3. generate a high quality answer based on the key information
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
        return json.loads(content)
    else:
        return f"Error: {response.status_code} - {response.text}"
    
def generate_rating(tokenized_string, question, answer):
    data = json.dumps({
        "model": "qwen/qwq-32b:free",
        "messages": [
            {
                "role": "user",
                "content": f"""Rank the 3 question and answers pair from 1 to 3 based on the quality of the question and answer. 
                where 3 is the highest and 1 is the lowest.
                Questions: {question}
                Answers: {answer} based on the text: {tokenized_string}
                Please follow the steps below to process the climate report snippet:
    1. read and understand the report carefully
    2. identify the key weather/climate related information in the report
    3. generate a high quality rating based on the key information
    4. use logical reasoning to arrive at the correct answer
    5. The final output must strictly use the following JSON format :("ranking": "Ranking from 1 to 3"
    Make sure to use double quotes for all keys and string values
    Note: Only output the final JSON
                """
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
        return json.loads(content)
    else:
        return f"Error: {response.status_code} - {response.text}"


question_output = generate_question(tokenized_string, 3, 1)
print(question_output)
question = question_output["Question"]

answer_output = generate_answer(tokenized_string, question)
print(answer_output)
answer = answer_output["answer"]

question = ["How do CMIP6 advancements and persistent biases impact confidence in attributing anthropogenic warming?",
            "How do CMIP6 models enhance attribution of human-induced warming over CMIP5 despite persistent regional biases?",
            "What are the implications of CMIP6 advancements in attributing anthropogenic warming despite persistent biases?",
            "What are the improvements in CMIP6 models over CMIP5 in replicating historical temperature patterns and alignment with paleoclimate reconstructions?",
            "What are the challenges and improvements in CMIP6 models over CMIP5 in replicating historical temperature patterns and alignment with paleoclimate reconstructions?"]


answer = ["CMIP6 advancements enhance confidence in attributing anthropogenic warming despite persistent biases.",
            "CMIP6 models improve attribution of human-induced warming over CMIP5 despite regional biases.",
            """CMIP6 models show improvements over CMIP5 in replicating historical temperature patterns, 
            better alignment with paleoclimate reconstructions (e.g., Pliocene and Eocene), and reduced 
            biases through enhanced model resolution. Persistent biases remain in high-latitude regions 
            like the Arctic, aerosol radiative forcing, and processes such as cloud physics, ocean circulation, 
            and surface energy budgets. Specific regional discrepancies include underestimated Arctic 
            warming during the mid-Holocene and historical period mismatches (e.g., mid-20th century 
            cooling and late 20th-century warming trends). Challenges in representing internal variability 
            (e.g., ENSO, AMO) and aerosol-cloud interactions also persist.""",
            "Human activities drive most observed warming, with climate models confirming dominant anthropogenic influence despite lingering regional biases and uncertainties.",
            "CMIP6 improves detection via refined global patterns, advanced aerosol modeling, and enhanced uncertainty analysis, despite lingering regional biases.",]

rating_output = generate_rating(tokenized_string, question, answer)
print(rating_output)
rating = rating_output["rating"]

print(rating)



