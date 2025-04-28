# run the train script on the superpod

import logging
import os
import socket
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# IMPORTANT: Set these variables BEFORE importing any HuggingFace libraries
hostname = socket.gethostname()
# Change to permanent cache location instead of temporary directory
cache_dir = "/home/htleungav/hf_cache"

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
    DataCollatorForLanguageModeling,
    set_seed,
)
import json
from datasets import Dataset
import torch
from peft import get_peft_model, LoraConfig, TaskType

def get_free_space(path) -> float:
    """Get free space in GB"""
    statvfs = os.statvfs(path)
    return (statvfs.f_bavail * statvfs.f_frsize) / (1024 ** 3)

def load_data(file_path):
    logger.info(f"Loading data from {file_path}")
    with open(file_path, 'r') as f:
        return json.load(f)

def prepare_datasets(train_file, val_file, tokenizer, max_length):
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
        texts = [prompt + completion + tokenizer.eos_token for prompt, completion in zip(prompts, completions)] # Add EOS token
        
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
    # Check free space on the cache directory
    free_space = get_free_space(cache_dir)
    logger.info(f"Free space in cache directory: {free_space:.2f} GB")
    if free_space < 10:  # Check if there's at least 10 GB free
        logger.warning("Low disk space in cache directory. Please free up some space.")
        return

    # Configuration
    base_model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"
    train_file = '/home/htleungav/UROP_IPCC_LLM/data/processed/test.json'
    val_file = '/home/htleungav/UROP_IPCC_LLM/data/processed/train.json'
    output_dir = '/home/htleungav/UROP_IPCC_LLM/lora_output/'
    
    logger.info(f"Initializing fine-tuning with base model: {base_model_name}")
    logger.info(f"Output will be saved to: {output_dir}")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    set_seed(42)  # Set seed for reproducibility
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name, 
        cache_dir=cache_dir  # Explicitly specify cache directory
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        use_safetensors=True,
        torch_dtype=torch.bfloat16,  # Use bfloat16 for better GPU performance
        device_map="auto",           # Let the library handle GPU mapping
        cache_dir=cache_dir,
        )

    # =======================  Apply LoRA to the model =========================
    # Inspect model architecture to find proper target modules
    linear_layers = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            linear_layers.append(name)
    
    logger.info("Available Linear layers in model:")
    for layer in linear_layers[:10]:  # Print first 10 to avoid flooding logs
        logger.info(f"Layer: {layer}")
    
    # Find attention projection layers - these typically contain 'q_proj', 'k_proj', 'v_proj', 'out_proj'
    attn_layers = [layer for layer in linear_layers if any(x in layer for x in ['q_proj', 'k_proj', 'v_proj', 'out_proj'])]
    
    if not attn_layers:
        # Fallback: target all linear layers that have 'down' or 'up' in transformer MLP blocks
        attn_layers = [layer for layer in linear_layers if any(x in layer for x in ['down', 'up', 'gate'])]
        if not attn_layers:
            # Last resort: use all linear layers (may increase training parameters significantly)
            attn_layers = linear_layers
    
    logger.info(f"Using target modules: {attn_layers[:5]}... (total: {len(attn_layers)})")
    
    # Apply LoRA to the model
    logger.info("Applying LoRA configuration")
    lora_config = LoraConfig(
        r=16,                  # Larger rank for better adaptation
        lora_alpha=32,         # Standard alpha
        lora_dropout=0.1,      # Add some dropout
        target_modules=attn_layers,  # Use detected modules
        bias="none",           # Explicitly set bias to none
        task_type=TaskType.CAUSAL_LM,  # Explicitly set the task type
    )
    # =======================================================================
    
    # Prepare datasets
    train_dataset, val_dataset = prepare_datasets(train_file, val_file, tokenizer, max_length=4096)
    
    # Data collator, allow automatic padding, attention mask, and other preprocessing tasks
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # We're not using masked language modeling 
    )
    
    train_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        
        # Enable better multi-GPU training
        deepspeed=None,  # Let HF detect optimal config
            
        # Full training parameters
        num_train_epochs=3,             # Run for full 3 epochs
        learning_rate=5e-5,             # Standard learning rate for fine-tuning
        weight_decay=0.01,              # Add weight decay for regularization
        lr_scheduler_type="cosine",     # Better scheduler for full training
        warmup_ratio=0.1,               # Add warmup steps
        
        # GPU optimization
        per_device_train_batch_size=8,  # Increase batch size for GPU
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,  # Adjust based on GPU memory
        gradient_checkpointing=True,    # Save memory
        
        # Enable mixed precision
        bf16=True,                  # Use bfloat16 for better performance
        fp16=False,                 # Disable fp16 for now
        
        # Enable proper checkpointing
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,             # Keep only last 3 checkpoints
        
        # Evaluation and logging
        eval_strategy="steps",  # Try this alternate naming
        eval_steps=200,

        # Logging 
        logging_steps=50,
        report_to="tensorboard",
        
        # Distributed training
        ddp_find_unused_parameters=False,
        dataloader_num_workers=4,       # Increase for faster data loading
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator, 
        tokenizer=tokenizer, 
    )

    model_data = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            model_data[name] = (module.in_features, module.out_features)

    logger.info("Linear layers in model:")
    for name, (in_f, out_f) in model_data.items():
        logger.info(f"Layer: {name}, Shape: {in_f}x{out_f}")

    
    # Start training
    logger.info("Starting training")

    checkpoint_dir = f"{output_dir}/checkpoint"
    if os.path.exists(checkpoint_dir):
        checkpoint_dirs = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")]
        if checkpoint_dirs:
            # Sort checkpoints by step number and resume from the latest one
            latest_checkpoint = sorted(checkpoint_dirs, key=lambda x: int(x.split("-")[1]), reverse=True)[0]
            checkpoint_dir = os.path.join(checkpoint_dir, latest_checkpoint)
            
            logger.info(f"Resuming training from checkpoint: {latest_checkpoint}")
            trainer.train(resume_from_checkpoint=checkpoint_dir)
        else:
            logger.info("No checkpoint found. Starting training from scratch.")
            trainer.train()
    else:
        logger.info("No checkpoint directory found. Starting training from scratch.")
        trainer.train()
        
    # Save the model
    logger.info(f"Saving model to {output_dir}/final-model")
    model.save_pretrained(f"{output_dir}/final-model")
    tokenizer.save_pretrained(f"{output_dir}/final-model")
    
    logger.info("Training completed successfully")

if __name__ == "__main__":
    main()