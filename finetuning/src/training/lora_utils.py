# This script is used to prepare the configuration JSON for the LoRA model and training configurations.
from typing import Dict, Any

def prepare_lora_config(base_model: str, lora_rank: int, lora_alpha: int, lora_dropout: float, trainable_layers: str) -> Dict[str, Any]:
    return {
        "base_model": base_model,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "trainable_layers": trainable_layers
    }

def prepare_training_config(learning_rate: float, batch_size: int, num_epochs: int, gradient_accumulation_steps: int, max_seq_length: int, evaluation_steps: int, save_steps: int, output_dir: str) -> Dict[str, Any]:
    return {
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "max_seq_length": max_seq_length,
        "evaluation_steps": evaluation_steps,
        "save_steps": save_steps,
        "output_dir": output_dir
    }

def save_config_to_json(config: Dict[str, Any], filepath: str) -> None:
    import json
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=4)

def main():
    lora_config = prepare_lora_config("Deepseek R1 1.5B", 16, 32, 0.1, "all")
    training_config = prepare_training_config(5e-5, 16, 3, 2, 512, 100, 500, "./output")

    save_config_to_json(lora_config, "./configs/lora_config.json")
    save_config_to_json(training_config, "./configs/training_config.json")

if __name__ == "__main__":
    main()