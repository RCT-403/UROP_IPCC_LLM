import os
import socket
import logging
import torch
import random
import numpy as np
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

# Test the trained model on SuperPod environment

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# IMPORTANT: Set cache directory BEFORE importing any HuggingFace libraries
hostname = socket.gethostname()
cache_dir = "/home/htleungav/hf_cache"  # Modify this path as needed for SuperPod

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

# SuperPod specific configurations
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # Adjust based on available GPUs

def set_seed(seed):
    """Set all seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_full_model(model_path):
    """Build a full model from the given path."""
    logger.info(f"Loading fine-tuned model from: {model_path}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,  # Use bfloat16 for better performance on SuperPod
        cache_dir=cache_dir,
        local_files_only=False,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        cache_dir=cache_dir
    )

    logger.info("Fine-tuned model and tokenizer loaded successfully")
    return model, tokenizer

def build_vanilla_model(base_model_name):
    """Build the vanilla model by loading the base model."""
    logger.info(f"Loading vanilla model: {base_model_name}")
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,  # Use bfloat16 for better performance on SuperPod
        trust_remote_code=True,
        cache_dir=cache_dir,
        local_files_only=False,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name, 
        trust_remote_code=True,
        cache_dir=cache_dir
    )

    logger.info("Vanilla model and tokenizer loaded successfully")
    return model, tokenizer

def generate_text(model, tokenizer, prompt, max_length=512, deterministic=False):
    """Generate text from the model given a prompt."""
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    start_time = time.time()
    with torch.no_grad():
        if deterministic:
            # Deterministic generation settings
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
                num_beams=5,
                early_stopping=True,
                no_repeat_ngram_size=3,
                repetition_penalty=1.2,
                length_penalty=1.0,
            )
        else:
            # Creative generation settings
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
    
    generation_time = time.time() - start_time
    
    # Decode and return the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response, generation_time

def test_model(model, tokenizer, test_prompts, deterministic, model_name):
    results = []
    
    logger.info(f"Testing {model_name} model with {'deterministic' if deterministic else 'creative'} generation")
    
    for i, prompt in enumerate(test_prompts, 1):
        logger.info(f"\n=== Prompt {i}: ===")
        logger.info(prompt)
        
        response, gen_time = generate_text(model, tokenizer, prompt, max_length=1024, deterministic=deterministic) 
        
        logger.info(f"\n=== Response (generated in {gen_time:.2f}s): ===")
        logger.info(response)
        logger.info(f"\n{'='*60}\n")
        
        results.append({
            "prompt": prompt,
            "response": response,
            "generation_time": gen_time
        })
    
    return results

def compare_models(vanilla_results, finetuned_results):
    """Compare results between vanilla and fine-tuned models"""
    logger.info("\n=== Model Comparison ===")
    
    for i in range(len(vanilla_results)):
        logger.info(f"\nPrompt {i+1}: {vanilla_results[i]['prompt']}")
        logger.info(f"\nVanilla model ({vanilla_results[i]['generation_time']:.2f}s):\n{vanilla_results[i]['response']}")
        logger.info(f"\nFine-tuned model ({finetuned_results[i]['generation_time']:.2f}s):\n{finetuned_results[i]['response']}")
        logger.info(f"\n{'='*60}")

def save_results_to_file(vanilla_results, finetuned_results, output_path="model_comparison_results.txt"):
    """Save the comparison results to a file"""
    with open(output_path, "w") as f:
        f.write("=== Model Comparison Results ===\n\n")
        
        for i in range(len(vanilla_results)):
            f.write(f"Prompt {i+1}: {vanilla_results[i]['prompt']}\n\n")
            f.write(f"Vanilla model ({vanilla_results[i]['generation_time']:.2f}s):\n{vanilla_results[i]['response']}\n\n")
            f.write(f"Fine-tuned model ({finetuned_results[i]['generation_time']:.2f}s):\n{finetuned_results[i]['response']}\n\n")
            f.write(f"{'='*60}\n\n")
    
    logger.info(f"Results saved to {output_path}")

if __name__ == "__main__":
    base_model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"
    finetuned_model_path = "/home/htleungav/UROP_IPCC_LLM/lora_output/final-model"  # Path to your fine-tuned model
    output_path = "/home/htleungav/UROP_IPCC_LLM/model_comparison_results.txt"
    
    set_seed(42)
    
    test_prompts = [
        "USER: What did the SROCC report conclude about global mean sea level rise rates between 2006-2015? Include the confidence levels and numerical values mentioned in the IPCC report.\nASSISTANT:",
        
        "USER: According to IPCC data, how much higher was the global mean sea level during the Last Interglacial period compared to today? Provide the confidence level for this estimate.\nASSISTANT:",
        
        "USER: Extract the specific sea level rise measurements from the IPCC reports for different time periods: 1902-2010, 1970-2015, 1993-2015, and 2006-2015.\nASSISTANT:"
    ]
    
    try:
        # Test both models and compare
        logger.info("Loading vanilla model...")
        vanilla_model, vanilla_tokenizer = build_vanilla_model(base_model_name)
        vanilla_results = test_model(vanilla_model, vanilla_tokenizer, test_prompts, deterministic=True, model_name="Vanilla")
        
        # Clear CUDA cache to free up memory
        torch.cuda.empty_cache()
        
        logger.info("Loading fine-tuned model...")
        # Load as a full model directly (not as LoRA)
        finetuned_model, finetuned_tokenizer = build_full_model(finetuned_model_path)
        finetuned_results = test_model(finetuned_model, finetuned_tokenizer, test_prompts, deterministic=True, model_name="Fine-tuned")
        
        # Compare the results
        compare_models(vanilla_results, finetuned_results)
        
        # Save results to file
        save_results_to_file(vanilla_results, finetuned_results, output_path)
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        import traceback
        logger.error(traceback.format_exc())