
import logging
import os
import socket
import tempfile
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# IMPORTANT: Set these variables BEFORE importing any HuggingFace libraries
hostname = socket.gethostname()
temp_dir = f"/tmp/hf_cache_{os.getenv('USER')}_{hostname}"

# Clean up any existing directory to start fresh
if os.path.exists(temp_dir):
    shutil.rmtree(temp_dir)
os.makedirs(temp_dir, exist_ok=True)
logger.info(f"Created custom cache directory: {temp_dir}")

# Set ALL possible cache-related environment variables
os.environ["TRANSFORMERS_CACHE"] = temp_dir
os.environ["HF_HOME"] = temp_dir
os.environ["HF_DATASETS_CACHE"] = temp_dir
os.environ["HUGGINGFACE_HUB_CACHE"] = temp_dir
os.environ["HF_MODULES_CACHE"] = temp_dir
os.environ["XDG_CACHE_HOME"] = temp_dir

from transformers import (
    Trainer, 
    TrainingArguments, 
    AutoModelForCausalLM, 
    AutoTokenizer,
    DataCollatorForLanguageModeling
)
import json
from datasets import Dataset
import torch
from peft import get_peft_model, LoraConfig, TaskType

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
    base_model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" 
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
        # device_map="auto",
        load_in_8bit=False, # make sure we don't use quantization
        use_safetensors=True,
        # torch_dtype=torch.float32,
        torch_dtype=torch.bfloat16,
    )

    # explicitly map the model to the GPU if available
    if torch.cuda.is_available():
        model = model.to("cuda")
        logger.info("Model moved to GPU")
    else:
        logger.info("No GPU available, using CPU")
    
    # Apply LoRA to the model
    logger.info("Applying LoRA configuration")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,                  # Rank dimension
        lora_alpha=32,         # Alpha parameter for LoRA scaling
        lora_dropout=0.05,     # Dropout probability for LoRA layers
        bias="none",           # Bias type
        target_modules=[       # Modules to apply LoRA to
            "q_proj",  # Query projection
            "k_proj",   # Key projection
            "v_proj",  # Value projection
            "o_proj",  # Output projection
            "gate_proj",  # Gate projection
            "down_proj", # Down projection
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

        # evaluation_strategy="steps", # this is in a new version of transformers
        # eval_steps=100,
        # save_strategy="steps",

        save_steps=200,
        save_total_limit=2,
        learning_rate=2e-4,
        weight_decay=0.01,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        warmup_steps=100,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        fp16=True,
        # report_to="none"
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        # tokenizer=tokenizer,
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