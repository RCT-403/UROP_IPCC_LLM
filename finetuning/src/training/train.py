import logging
import os
import socket
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# IMPORTANT: Set these variables BEFORE importing any HuggingFace libraries
hostname = socket.gethostname()
# Change to permanent cache location instead of temporary directory
cache_dir = f"/disk/r089/htleungav/hf_cache"

# Create the cache directory if it doesn't exist, but don't delete existing one
os.makedirs(cache_dir, exist_ok=True)
logger.info(f"Using cache directory: {cache_dir}")

# Set ALL possible cache-related environment variables
os.environ["TRANSFORMERS_CACHE"] = cache_dir
os.environ["HF_HOME"] = cache_dir
os.environ["HF_DATASETS_CACHE"] = cache_dir
os.environ["HUGGINGFACE_HUB_CACHE"] = cache_dir
os.environ["HF_MODULES_CACHE"] = cache_dir
os.environ["XDG_CACHE_HOME"] = cache_dir

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
    base_model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"
    train_file = '/home/htleungav/UROP_IPCC_LLM/data/processed/test.json'
    val_file = '/home/htleungav/UROP_IPCC_LLM/data/processed/train.json'
    # Store output in the linked directory
    output_dir = '/disk/r089/htleungav/model_outputs'
    
    logger.info(f"Initializing fine-tuning with base model: {base_model_name}")
    logger.info(f"Output will be saved to: {output_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name, 
        cache_dir=cache_dir  # Explicitly specify cache directory
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        load_in_8bit=False,
        use_safetensors=True,
        torch_dtype=torch.float32,  # Use float32 instead of bfloat16 for CPU
        cache_dir=cache_dir,  # Explicitly specify cache directory
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
        r=4,                  # Smaller rank dimension
        lora_alpha=16,        # Smaller alpha
        lora_dropout=0.0,     # No dropout for testing
        target_modules=["q_proj", "v_proj"]  # Target fewer modules
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Prepare datasets
    train_dataset, val_dataset = prepare_datasets(train_file, val_file, tokenizer)
    # For testing, use only a small subset of data
    train_dataset = train_dataset.select(range(min(30, len(train_dataset)))) # choose the first 30 samples
    val_dataset = val_dataset.select(range(min(15, len(val_dataset))))
    
    # Data collator, allow automatic padding, attention mask, and other preprocessing tasks
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # We're not using masked language modeling
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,

        # Reduce evaluation frequency for faster testing
        eval_strategy="steps", 
        eval_steps=10,           # Evaluate less frequently
        save_strategy="no",       # Don't save checkpoints for testing
        
        # Minimal learning parameters for testing
        learning_rate=2e-4,       # Slightly higher learning rate for faster convergence
        weight_decay=0.0,         # Remove weight decay for faster computation
        lr_scheduler_type="linear", # Simpler scheduler
        
        # Reduce batch sizes for CPU training
        per_device_train_batch_size=2, # Smaller batch size for CPU
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8, # Accumulate more gradients instead
        num_train_epochs=0.1,     # Just 10% of one epoch for testing
        max_steps=10,             # Or limit to a fixed number of steps
        
        # Disable features that slow down CPU training
        fp16=False,               # No mixed precision on CPU
        fp16_opt_level=None,      # Remove this parameter
        dataloader_num_workers=0, # No parallelization for testing
        group_by_length=False,    # Disable grouping to simplify
        
        # Minimal logging
        logging_steps=5,
        report_to="none",         # Disable reporting for testing
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