"""
Конфигурация для fine-tuning Qwen3-14B
"""
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    model_name: str = "unsloth/Qwen3-14B-Base-unsloth-bnb-4bit"
    max_seq_length: int = 2048
    dtype: Optional[str] = None  # auto detection
    load_in_4bit: bool = True

@dataclass
class LoraConfig:
    r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0
    target_modules: tuple = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    )
    bias: str = "none"
    use_gradient_checkpointing: str = "unsloth"
    random_state: int = 3407
    use_rslora: bool = False

@dataclass
class TrainingConfig:
    output_dir: str = "outputs"
    per_device_train_batch_size: int = 1  # уменьшено для 16GB VRAM
    gradient_accumulation_steps: int = 8  # эффективный batch = 8
    warmup_ratio: float = 0.03
    num_train_epochs: int = 1
    learning_rate: float = 2e-4
    logging_steps: int = 10
    eval_strategy: str = "no"  # отключено - eval жрёт память
    eval_steps: int = 500
    save_strategy: str = "steps"
    save_steps: int = 500
    optim: str = "adamw_8bit"
    weight_decay: float = 0.001
    lr_scheduler_type: str = "linear"
    seed: int = 3407
    report_to: str = "none"
    fp16: bool = False
    bf16: bool = True

@dataclass
class DataConfig:
    # Пути к датасетам
    finetune_train: str = "finetune_dataset/train.jsonl"
    finetune_val: str = "finetune_dataset/validation.jsonl"
    data_final_pars: str = "data_final_pars-20251221T120104Z-1-001/data_final_pars/training_data.jsonl"
    
    # Объединенные выходные файлы
    combined_train: str = "combined_data/train.jsonl"
    combined_val: str = "combined_data/validation.jsonl"
    
    # Доля валидации для data_final_pars (т.к. там нет отдельного val)
    val_split_ratio: float = 0.1

# Промпт-шаблон в стиле Alpaca
ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

