from datasets import load_dataset, DatasetDict, Dataset

from transformers import (
    AutoTokenizer,
    AutoConfig, 
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer)

from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig
import evaluate
import torch
import numpy as np
import json

with open("dataset.txt", "r") as file:
    data = json.loads(file.read())

custom_dataset = Dataset.from_dict(data)
print(custom_dataset)

# Combine Question and Answer into a single input
def preprocess_data(example):
    example["text"] = example["Question"] + " [SEP] " + example["Answer"]
    return example

# Apply preprocessing
custom_dataset = custom_dataset.map(preprocess_data)

# Split the dataset into train and test sets
split_dataset = custom_dataset.train_test_split(test_size=0.2, seed=42)

# Load a tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Tokenize the dataset
def tokenize_function(example):
    return {
        "input_ids": tokenizer(
            example["text"],
            padding="max_length",
            truncation=True,
            max_length=512
        )["input_ids"],
        "attention_mask": tokenizer(
            example["text"],
            padding="max_length",
            truncation=True,
            max_length=512
        )["attention_mask"],
        "labels": example["Rating"],  # Add the Rating column as labels
    }

# Tokenize the dataset
tokenized_dataset = split_dataset.map(tokenize_function, batched=True)

tokenized_dataset = split_dataset.map(tokenize_function, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=5  # Replace with the number of unique ratings
)

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=1)
model.config.problem_type = "regression"

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy" if model.config.problem_type != "regression" else "mse",
)

from sklearn.metrics import accuracy_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}

from sklearn.metrics import mean_squared_error

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.squeeze()
    return {"mse": mean_squared_error(labels, predictions)}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()


# Move the model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define a function to test the model
def test_model(question, answer):
    # Combine Question and Answer into a single input
    text = question + " [SEP] " + answer

    # Tokenize the input
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512
    ).to(device)

    # Get predictions
    with torch.no_grad():
        logits = model(**inputs).logits

    # For regression, return the raw score
    if model.config.problem_type == "regression":
        prediction = logits.squeeze().item()
        print(f"Predicted Rating (Regression): {prediction}")
    else:  # For classification, return the class label
        prediction = torch.argmax(logits, dim=1).item()
        print(f"Predicted Rating (Classification): {prediction}")

# Example Question and Answer
question = "How do CMIP6 models improve upon CMIP5?"
answer = "its better"

# Test the model
test_model(question, answer)
