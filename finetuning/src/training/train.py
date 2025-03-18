from transformers import (
    Trainer, 
    TrainingArguments, 
    AutoModelForCausalLM, 
    AutoTokenizer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import json
import os
import torch
from peft import get_peft_model, LoraConfig, TaskType
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(file_path):
    logger.info(f"Loading data from {file_path}")
    with open(file_path, 'r') as f:
        return json.load(f)

def prepare_datasets(train_file, val_file, tokenizer, max_length=1024):
    """Prepare datasets in the correct format for training"""
    logger.info("Preparing datasets for training")
    
    # Load data from JSON files
    train_data = load_data(train_file)
    val_data = load_data(val_file)
    
    # Convert to datasets format
    def convert_to_dataset(examples):
        prompts = [item["prompt"] for item in examples]
        completions = [item["completion"] for item in examples]
        
        # Combine prompt and completion for training
        texts = [prompt + completion for prompt, completion in zip(prompts, completions)]
        
        return {"text": texts}
    
    train_dataset = Dataset.from_dict(convert_to_dataset(train_data))
    val_dataset = Dataset.from_dict(convert_to_dataset(val_data))
    
    # Tokenize function
    def tokenize_function(examples):
        outputs = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt"
        )
        # For causal language modeling, labels are the same as inputs
        outputs["labels"] = outputs["input_ids"].clone()
        return outputs
    
    # Apply tokenization
    tokenized_train = train_dataset.map(
        tokenize_function, 
        batched=True,
        remove_columns=["text"]
    )
    
    tokenized_val = val_dataset.map(
        tokenize_function, 
        batched=True,
        remove_columns=["text"]
    )
    
    logger.info(f"Prepared {len(tokenized_train)} training examples and {len(tokenized_val)} validation examples")
    
    return tokenized_train, tokenized_val

def main():
    # Configuration
    base_model_name = "deepseek-ai/deepseek-coder-1.3b-base" 
    train_file = './data/processed/train.json'
    val_file = './data/processed/test.json'
    output_dir = './output'
    
    logger.info(f"Initializing fine-tuning with base model: {base_model_name}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float32,
        device_map="auto"
    )
    
    # Apply LoRA to the model
    logger.info("Applying LoRA configuration")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,                  # Rank dimension
        lora_alpha=32,         # Alpha parameter for LoRA scaling
        lora_dropout=0.05,     # Dropout probability for LoRA layers
        bias="none",           # Bias type
        target_modules=[       # Modules to apply LoRA to
            "q_proj", 
            "k_proj", 
            "v_proj", 
            "o_proj", 
            "gate_proj", 
            "down_proj", 
            "up_proj"
        ]
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Prepare datasets
    train_dataset, val_dataset = prepare_datasets(train_file, val_file, tokenizer)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # We're not using masked language modeling
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        learning_rate=2e-4,
        weight_decay=0.01,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        warmup_steps=100,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        fp16=True,
        report_to="none"
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Start training
    logger.info("Starting training")
    trainer.train()
    
    # Save the model
    logger.info(f"Saving model to {output_dir}/final-model")
    model.save_pretrained(f"{output_dir}/final-model")
    tokenizer.save_pretrained(f"{output_dir}/final-model")
    
    logger.info("Training completed successfully")

if __name__ == "__main__":
    main()

    # paths = [
    #     './data/processed/training_examples_2.json',
    #     './data/processed/training_examples_3.json',
    #     './data/processed/training_examples_4.json',
    #     './data/processed/training_examples_11.json'
    # ]
    # combine_chapters(paths, './data/processed/training_examples_combined.json')