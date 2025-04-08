# Test the trained lora

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import os

def build_lora_model(base_model_name, adapter_path):
    """Build the LoRA model by loading the base model and adapter."""
    assert os.path.exists(adapter_path), f"Adapter path {adapter_path} does not exist."

    # Load the configuration
    config = PeftConfig.from_pretrained(adapter_path)
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float32,
        device_map="auto"
    )
    
    # Load LoRA adapter
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, adapter_path)
     
    return model

def build_vanilla_model(base_model_name):
    """Build the vanilla model by loading the base model."""
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    return model, tokenizer

def generate_text(model, tokenizer, prompt, max_length=512):
    """Generate text from the model given a prompt.
    remark: the 512 token is approxmately 350 to 400 words.
    """
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    
    # Generate output
    # print(f"Generating response for prompt: {prompt}")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            pad_token_id=tokenizer.eos_token_id, # Pad to EOS token

            # # apply sampling, which consider a distribution of words
            do_sample=True,
            top_p=0.92, # control the diversity of the output (0.0: no randomness, 1.0: maximum randomness)
            temperature=0.7, # control the randomness of the output (0.0: deterministic, 1.0: maximum randomness)

            repetition_penalty=1.2,  # Penalize repetition
            no_repeat_ngram_size=3,  # Avoid repeating 3-grams
            
            # improve the quality of the output
            num_beams=5,  # add beam search
            length_penalty=1.0,  # control the length of the output
        )
    
    # Decode and return the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def test_model(model, tokenizer, test_prompts):
    for prompt in test_prompts:
        response = generate_text(model, tokenizer, prompt, max_length=512*4)
        print(response)

if __name__ == "__main__":
    # Load the fine-tuned model
    # adapter_path = "/home/htleungav/UROP_IPCC_LLM/output/final-model"
    # base_model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  
    base_model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    model, tokenizer = build_vanilla_model(base_model_name)
    print("Model and tokenizer loaded successfully")

    test_prompts = [
        "Summarize the key findings on climate change impacts from the IPCC report:",
        "What mitigation strategies does the IPCC recommend for reducing greenhouse gas emissions?",
        "Explain the projected sea level rise according to the IPCC:",
    ]
    
    # Test the model
    test_model(model, tokenizer, test_prompts)