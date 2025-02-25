import requests
import json

response = requests.post(
  url="https://openrouter.ai/api/v1/chat/completions",
  headers={
    "Authorization": "Bearer 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",
    "Content-Type": "application/json",
},    
  data=json.dumps({
    "model": "cognitivecomputations/dolphin3.0-mistral-24b:free",
    "messages": [
      {
        "role": "user",
        "content": "What is the meaning of life?"
      }
    ],

  })
)

# Print the response
if response.status_code == 200:
    # Successful response
    response_data = response.json()  # Parse JSON response
    print(json.dumps(response_data, indent=2))  # Pretty print the JSON response
else:
    # Print error response
    print(f"Error: {response.status_code} - {response.text}")
