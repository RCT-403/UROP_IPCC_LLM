from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load the saved model
loaded_model = AutoModelForSequenceClassification.from_pretrained("./saved_model")

# Load the saved tokenizer
loaded_tokenizer = AutoTokenizer.from_pretrained("./saved_model")

print("Model and tokenizer have been successfully loaded.")

# Example text
text = "This movie was fantastic!"

# Tokenize the input
inputs = loaded_tokenizer.encode(text, return_tensors="pt")

# Get predictions
logits = loaded_model(inputs).logits
predictions = torch.argmax(logits, dim=1)

print(f"Prediction: {predictions}")


