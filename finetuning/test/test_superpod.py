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

def generate_text(model, tokenizer, prompt: str, max_new_tokens: int, deterministic: bool):
    """Generate text from the model given a prompt."""
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    start_time = time.time()
    with torch.no_grad():
        if deterministic:
            # Deterministic generation settings
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
                num_beams=3, # reduce num_beams to 3 for faster generation
                early_stopping=True, 
                no_repeat_ngram_size=3,
                repetition_penalty=1.2,
                length_penalty=1.0,
            )
        else:
            # Creative generation settings
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens, # restrict num of tokens to generate
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
    

    input_length = inputs.input_ids.shape[1]
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True) # for debugging, see if the truncation work as expected 
    response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    
    logger.info(f"Generated tokens: {outputs[0].shape[0] - input_length}")
    logger.info(f"Full output length: {len(full_output)}")
    logger.info(f"Response length: {len(response)}")
    
    # If response is empty, return the full output (for debugging)
    if not response.strip():
        logger.warning("Empty response detected! Using full output for debugging.")
        return full_output, generation_time

    return response, generation_time

def test_model(model, tokenizer, test_prompts, deterministic, model_name):
    results = []
    
    logger.info(f"Testing {model_name} model with {'deterministic' if deterministic else 'creative'} generation")
    
    for i, prompt in enumerate(test_prompts, 1):
        logger.info(f"\n=== Prompt {i}: ===")
        logger.info(prompt)
        
        response, gen_time = generate_text(model, tokenizer, prompt, max_new_tokens=1024, deterministic=deterministic) 
        
        logger.info(f"\n=== Response (generated in {gen_time:.2f}s): ===")
        logger.info(response)
        logger.info(f"\n{'='*60}\n")
        
        results.append({
            "prompt": prompt,
            "response": response,
            "generation_time": gen_time
        })
    
    return results

'''
# ...existing code...
import logging # Make sure logging is imported
logger = logging.getLogger(__name__) # Ensure logger is defined

def generate_text(model, tokenizer, prompt: str, max_new_tokens: int, deterministic: bool, num_beams: int = 5, model_name: str = "Unknown"): # Add model_name parameter
    """Generate text from the model given a prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    generation_config = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
        "no_repeat_ngram_size": 3,
        "repetition_penalty": 1.2,
        "length_penalty": 1.0,
    }
    # ... (rest of generation_config setup) ...
    if deterministic:
        generation_config.update({
            "do_sample": False,
            "num_beams": num_beams,
            "early_stopping": False,
        })
    else: # Creative
        generation_config.update({
            "do_sample": True,
            "top_p": 0.92,
            "temperature": 0.7,
            "num_beams": num_beams,
        })


    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_config)
    generation_time = time.time() - start_time

    input_length = inputs.input_ids.shape[1]
    response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

    # Log raw tokens specifically for the problematic case
    if model_name == "Fine-tuned" and "AED and Actual Evapotranspiration" in prompt:
        logger.info(f"--- DEBUG: Raw output tokens for Prompt 2 (Fine-tuned) ---")
        logger.info(f"Input length: {input_length}")
        logger.info(f"Output sequence length: {len(outputs[0])}")
        logger.info(f"Generated token IDs: {outputs[0][input_length:].tolist()}")
        logger.info(f"EOS token ID: {tokenizer.eos_token_id}")
        logger.info(f"--- END DEBUG ---")

    return response, generation_time

def test_model(model, tokenizer, test_prompts, deterministic, model_name):
    results = []
    logger.info(f"Testing {model_name} model with {'deterministic' if deterministic else 'creative'} generation")

    for i, prompt in enumerate(test_prompts, 1):
        logger.info(f"\n=== Prompt {i}: ===")
        logger.info(prompt)

        current_num_beams = 5
        # No need to override num_beams=1 anymore based on last result
        # if deterministic and model_name == "Fine-tuned" and i == 2:
        #     logger.info(">>> Using default num_beams=5 for this prompt <<<")
        #     # current_num_beams = 1 # Reverted change

        response, gen_time = generate_text(
            model,
            tokenizer,
            prompt,
            max_new_tokens=1024,
            deterministic=deterministic,
            num_beams=current_num_beams,
            model_name=model_name # Pass model_name for debug logging
        )

        logger.info(f"\n=== Response (generated in {gen_time:.2f}s): ===")
        logger.info(response)
        logger.info(f"\n{'='*60}\n")

        results.append({
            "prompt": prompt,
            "response": response,
            "generation_time": gen_time
        })

    return results
'''

# ... (rest of the code) ...

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
    
    # test_prompts = [
    #     "USER: What did the SROCC report conclude about global mean sea level rise rates between 2006-2015? Include the confidence levels and numerical values mentioned in the IPCC report.",
    #     "USER: According to IPCC data, how much higher was the global mean sea level during the Last Interglacial period compared to today? Provide the confidence level for this estimate.",
    #     "USER: Extract the specific sea level rise measurements from the IPCC reports for different time periods: 1902-2010, 1970-2015, 1993-2015, and 2006-2015."
    # ]

    extra_info = "Atmospheric evaporative demand (AED) quantifies the maximum\namount of actual evapotranspiration (ET) that can happen from land\nsurfaces if they are not limited by water availability (Table 11.A.1).\nAED is affected by radiative and aerodynamic components. For this\nreason, the atmospheric dryness, often quantified with the relative\nhumidity or the vapour pressure deficit (VPD), is not equivalent to\nthe AED, as other variables are also highly relevant, including solar\nradiation and wind speed (Hobbins et al., 2012; McVicar et al., 2012a;\nSheffield et al., 2012). AED can be estimated using different methods\n(McMahon et al., 2013), and those solely based on air temperature\n(e.g., Hargreaves, Thornthwaite) usually overestimate it in terms of\nmagnitude and temporal trends (Sheffield et al., 2012), in particular,\nin the context of substantial background warming. Physically-based\ncombination methods such as the Penman-Monteith equation are\nmore adequate and recommended since 1998 by the United Nations\nFood and Agriculture Oganization (Pereira et al., 2015). For this reason,\nthe assessment of this Chapter, when considering atmosphericbased drought indices, only includes AED estimates using the latter\n(see also Section 11.9). AED is generally higher than ET, since AED\nrepresents an upper bound for ET. Hence, an AED increase does not\nnecessarily lead to increased ET (Milly and Dunne, 2016), in particular\nunder drought conditions given soil moisture limitation (Bonan et al.,\n2014; Berg et al., 2016; Konings et al., 2017; Stocker et al., 2018).\nIn general, AED is highest in regions where ET is lowest (e.g., desert\nareas), further illustrating the decoupling between the two variables\nunder limited soil moisture.\n\nThe influence of AED on drought depends on the drought type,\nbackground climate, the environmental conditions and the moisture\navailability (Hobbins et al., 2016, 2017; Vicente-Serrano et al.,\n2020a). This influence also includes effects not related to increased\nET. Under low soil moisture conditions, increased AED increases\nplant stress, enhancing the severity of agricultural and ecological\ndroughts (Williams et al., 2013; Allen et al., 2015; McDowell et al.,\n2016; Grossiord et al., 2020). Moreover, high VPD impacts overall\nplant physiology; it affects the leaf and xylem safety margins, and\ndecreases the sap velocity and plant hydraulic conductance (Fontes\net al., 2018). VPD also affects the plant metabolism of carbon and,\nif prolonged, it may cause plant mortality via carbon starvation\n(Breshears et al., 2013; Hartmann, 2015). Drought projections based\n\n\nexclusively on AED metrics overestimate changes in soil moisture and\nrunoff deficits. Nevertheless, AED also directly impacts hydrological\ndrought, as ET from surface waters is not limited (Wurbs and Ayala,\n2014; Friedrich et al., 2018; Hogeboom et al., 2018; K. Xiao et al., 2018),\nand this effect increases under climate change projections (W. Wang\net al., 2018; Althoff et al., 2020). In addition, high AED increases crop\nwater consumptions in irrigated lands (Garc\u00eda-Gariz\u00e1bal et al., 2014),\ncontributing to intensifying hydrological droughts downstream (Fazel\net al., 2017; Vicente-Serrano et al., 2017).\n\nOn subseasonal to decadal scales, temporal variations in AED are\nstrongly controlled by circulation variability (Williams et al., 2014;\nChai et al., 2018; Martens et al., 2018), but thermodynamic processes\nalso play a fundamental role and, under human-induced climate\nchange, dominate the changes in AED. Atmospheric warming due\nto increased atmospheric CO2 concentrations increases AED by\nmeans of enhanced VPD in the absence of other influences (Scheff\nand Frierson, 2015). Because of the greater warming over land than\nover oceans (Sections 2.3.1.1 and 11.3), the saturation pressure of\nwater vapour increases more over land than over oceans; oceanic\nair masses advected over land thus contain insufficient water vapour\nto keep pace with the greater increase in saturation vapour pressure\nover land (Sherwood and Fu, 2014; Byrne and O\u2019Gorman, 2018;\nFindell et al., 2019). Land\u2013atmosphere feedbacks are also important\nin affecting atmospheric moisture content and temperature, with\nresulting effects on relative humidity and VPD (Box 11.1; Berg et al.,\n2016; Haslinger et al., 2019; S. Zhou et al., 2019)."
    
    if extra_info is None:
        test_prompts = [
            # "Prompt 1: Definition and Distinction"
            "USER: Define Atmospheric Evaporative Demand (AED). Explain what factors influence it and why it is distinct from metrics like relative humidity or vapour pressure deficit (VPD)." ,
            # "Prompt 2: Relationship with Actual Evapotranspiration (ET)"
            "USER: What is the relationship between AED and Actual Evapotranspiration (ET)? Does an increase in AED necessarily lead to an increase in ET? Explain the conditions under which they might be decoupled, using information solely from IPCC." ,
            # "Prompt 3: Influence on Drought and Estimation Methods"
            "USER: Describe how AED influences different types of drought (agricultural, ecological, hydrological) according IPCC report. Additionally, what method for estimating AED is recommended in IPCC, and why are temperature-based methods considered less suitable?" ,
        ]
    else: 
        # slightly modified prompts to ensure the model uses the extra information
        test_prompts = [
            # "Prompt 1: Definition and Distinction"
            "USER: Define Atmospheric Evaporative Demand (AED). Explain what factors influence it and why it is distinct from metrics like relative humidity or vapour pressure deficit (VPD)." ,
            # "Prompt 2: Relationship with Actual Evapotranspiration (ET)"
            "USER: What is the relationship between AED and Actual Evapotranspiration (ET)? Does an increase in AED necessarily lead to an increase in ET? Explain the conditions under which they might be decoupled, using information solely from the text." ,
            # "Prompt 3: Influence on Drought and Estimation Methods"
            "USER: Describe how AED influences different types of drought (agricultural, ecological, hydrological) according to the provided text. Additionally, what method for estimating AED is recommended in the text, and why are temperature-based methods considered less suitable?" ,
        ]

    for i, prompt in enumerate(test_prompts):
        original_prompt = prompt
        # Add extra information to the prompt
        test_prompts[i] = "User: I need information about Atmospheric Evaporative Demand (AED)." 
        
        test_prompts[i] += "\n\nHere is the text from IPCC report about AED:\n\n"
        test_prompts[i] += extra_info

        test_prompts[i] += "\n\nBased on this IPCC information, " + original_prompt[6:]  # Remove USER:

        test_prompts[i] += "\n\nAssistant:"
    
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