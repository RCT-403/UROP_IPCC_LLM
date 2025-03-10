from typing import Dict, Any

class TrainingConfig:
    def __init__(self):
        self.base_model = "Deepseek R1 1.5B"
        self.lora_rank = 16
        self.lora_alpha = 32
        self.lora_dropout = 0.1
        self.trainable_layers = "all"
        self.learning_rate = 5e-5
        self.batch_size = 16
        self.num_epochs = 3
        self.gradient_accumulation_steps = 2
        self.max_seq_length = 512
        self.evaluation_steps = 100
        self.save_steps = 500
        self.output_dir = "./output"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "base_model": self.base_model,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "trainable_layers": self.trainable_layers,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_seq_length": self.max_seq_length,
            "evaluation_steps": self.evaluation_steps,
            "save_steps": self.save_steps,
            "output_dir": self.output_dir,
        }