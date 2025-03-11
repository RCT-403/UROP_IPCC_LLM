from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer
import json
import os

def load_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def format_data(data):
    formatted_data = {}
    for section in data:
        title = section['title']
        content = section['content']
        formatted_data[title] = content
    return formatted_data

def main():
    # Load formatted data
    formatted_data_path = './data/processed/formatted_data.json'
    formatted_data = load_data(formatted_data_path)

    # Prepare training data
    train_data = format_data(formatted_data)

    # Load model and tokenizer
    model_name = "Deepseek R1 1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Training arguments
    training_args = TrainingArguments(
        output_dir='./output',
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=500,
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        num_train_epochs=3,
        gradient_accumulation_steps=2,
        max_steps=1000,
        logging_dir='./logs',
        logging_steps=100,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=train_data,  # Use a separate validation dataset if available
    )

    # Start training
    trainer.train()

if __name__ == "__main__":
    main()