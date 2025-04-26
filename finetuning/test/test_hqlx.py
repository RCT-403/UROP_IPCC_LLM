# Test the trained lora

import os
import socket
import logging
import torch
import random
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# IMPORTANT: Set cache directory BEFORE importing any HuggingFace libraries
hostname = socket.gethostname()
cache_dir = "/disk/r089/htleungav/hf_cache"

# Create the cache directory if it doesn't exist
os.makedirs(cache_dir, exist_ok=True)
logger.info(f"Using cache directory: {cache_dir}")

# Set ALL possible cache-related environment variables
os.environ["TRANSFORMERS_CACHE"] = cache_dir
os.environ["HF_HOME"] = cache_dir
os.environ["HF_DATASETS_CACHE"] = cache_dir
os.environ["HUGGINGFACE_HUB_CACHE"] = cache_dir
os.environ["HF_MODULES_CACHE"] = cache_dir
os.environ["XDG_CACHE_HOME"] = cache_dir

# Now import the rest of the libraries
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

def set_seed(seed):
    """Set all seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_lora_model(base_model_name, adapter_path):
    """Build the LoRA model by loading the base model and adapter."""
    assert os.path.exists(adapter_path), f"Adapter path {adapter_path} does not exist."

    # Load the configuration
    config = PeftConfig.from_pretrained(adapter_path)
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float32,
        device_map="auto",
        cache_dir=cache_dir,  # Explicitly specify cache directory
        local_files_only=False,  # Control whether to use local files only
    )
    
    # Load tokenizer - prioritize from adapter path first, fall back to base model
    print("Loading tokenizer...")
    try:
        # Try to load the tokenizer from the adapter path first
        tokenizer = AutoTokenizer.from_pretrained(
            adapter_path,
            cache_dir=cache_dir
        )
        print("Tokenizer loaded from adapter path")
    except Exception as e:
        print(f"Could not load tokenizer from adapter path: {e}")
        print("Loading tokenizer from base model instead")
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name, 
            cache_dir=cache_dir
        )
    
    # Load LoRA adapter
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, adapter_path)

    print("LoRA adapter loaded successfully")
    return model, tokenizer

def build_vanilla_model(base_model_name):
    """Build the vanilla model by loading the base model."""
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=cache_dir,  # Explicitly specify cache directory
        local_files_only=False,  # Control whether to use local files only
    )

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name, 
        trust_remote_code=True,
        cache_dir=cache_dir  # Explicitly specify cache directory
    )

    print("Model and tokenizer loaded successfully")
    return model, tokenizer

def generate_text(model, tokenizer, prompt, max_length=512, deterministic=False):
    """Generate text from the model given a prompt."""
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        if deterministic:
            # Deterministic generation settings
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,           # Don't sample - use greedy decoding
                num_beams=5,               # Still use beam search for better quality
                early_stopping=True,       # Stop when all beams reach EOS
                no_repeat_ngram_size=3,    # Can keep this
                repetition_penalty=1.2,    # Can keep this
                length_penalty=1.0,        # Can keep this
            )
        else:
            # Your current creative generation settings
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True, 
                top_p=0.92,
                temperature=0.7,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                num_beams=5,
                length_penalty=1.0,
            )
    
    # Decode and return the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def test_model(model, tokenizer, test_prompts, deterministic):
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n=== Prompt {i}: ===")
        print(prompt)
        print("\n=== Response: ===")
        response = generate_text(model, tokenizer, prompt, max_length=1024, deterministic=deterministic) 
        print(response)
        print(f"\n{'='*60}\n")


if __name__ == "__main__":
    base_model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"
    adapter_path = "/home/htleungav/r089/model_outputs/final-model" 

    set_seed(42)  # Or any other seed number

    model, tokenizer = build_vanilla_model(base_model_name)
    # model, tokenizer = build_lora_model(base_model_name, adapter_path)

    # test_prompts = [
    #     "Summarize the key findings on climate change impacts from the IPCC report:",
    #     "What mitigation strategies does the IPCC recommend for reducing greenhouse gas emissions?",
    #     "Explain the projected sea level rise according to the IPCC:",
    # ]
    # test_prompts = [
    # "USER: Summarize the key findings on climate change impacts from the IPCC report:\nASSISTANT:",
    # "USER: Please describe the mitigation strategies that the IPCC recommends for reducing greenhouse gas emissions.\nASSISTANT:",
    # "USER: Explain the projected sea level rise according to the IPCC:\nASSISTANT:"
    # ]
    test_prompts = [
    "USER: What did the SROCC report conclude about global mean sea level rise rates between 2006-2015? Include the confidence levels and numerical values mentioned in the IPCC report.\nASSISTANT:",
    
    "USER: According to IPCC data, how much higher was the global mean sea level during the Last Interglacial period compared to today? Provide the confidence level for this estimate.\nASSISTANT:",
    
    "USER: Extract the specific sea level rise measurements from the IPCC reports for different time periods: 1902-2010, 1970-2015, 1993-2015, and 2006-2015.\nASSISTANT:"
    ]
    
    # Test the model
    test_model(model, tokenizer, test_prompts, deterministic=True)