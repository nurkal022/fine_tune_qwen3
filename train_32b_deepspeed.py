"""
Fine-tuning Qwen3-32B с DeepSpeed ZeRO-3
Распределение модели между 2x RTX 5090 (64GB суммарно)
БЕЗ квантизации - используем FP16/BF16
"""
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
import deepspeed

# ============== КОНФИГУРАЦИЯ ==============
MODEL_NAME = "Qwen/Qwen2.5-32B"
MAX_SEQ_LENGTH = 2048
OUTPUT_DIR = "outputs_32b_deepspeed"

# LoRA параметры
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Training
BATCH_SIZE = 1  # на GPU
GRAD_ACCUM = 8  # эффективный batch = 8 * 2 GPU = 16
NUM_EPOCHS = 1
LEARNING_RATE = 1e-4
SAVE_STEPS = 500
LOGGING_STEPS = 10

# Data
TRAIN_FILE = "combined_data/train.jsonl"
VAL_FILE = "combined_data/validation.jsonl"

# Prompt template
ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""


def print_gpu_info():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            total = props.total_memory / 1024**3
            print(f"GPU {i}: {props.name} | {allocated:.1f}GB allocated / {total:.0f}GB total")


def main():
    print("=" * 60)
    print("Fine-tuning Qwen3-32B с DeepSpeed ZeRO-3")
    print("Распределение между 2x RTX 5090 (64GB)")
    print("=" * 60)
    print_gpu_info()

    # 1. Загрузка модели БЕЗ квантизации
    print("\n[1/6] Загрузка модели (FP16, распределение между GPU)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",  # автоматическое распределение
        dtype=torch.bfloat16,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    print_gpu_info()

    # 2. LoRA
    print("\n[2/6] Применение LoRA...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 3. Датасет
    print("\n[3/6] Загрузка датасета...")
    
    def formatting_func(examples):
        texts = []
        for instr, inp, out in zip(examples["instruction"], examples["input"], examples["output"]):
            text = ALPACA_PROMPT.format(instr, inp, out) + tokenizer.eos_token
            texts.append(text)
        return {"text": texts}

    dataset = load_dataset("json", data_files={"train": TRAIN_FILE, "validation": VAL_FILE})
    train_dataset = dataset["train"].map(formatting_func, batched=True, remove_columns=dataset["train"].column_names)
    
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Effective batch size: {BATCH_SIZE * GRAD_ACCUM * torch.cuda.device_count()}")

    # 4. DeepSpeed конфиг
    print("\n[4/6] Настройка DeepSpeed ZeRO-3...")
    ds_config = {
        "train_batch_size": BATCH_SIZE * GRAD_ACCUM * torch.cuda.device_count(),
        "train_micro_batch_size_per_gpu": BATCH_SIZE,
        "gradient_accumulation_steps": GRAD_ACCUM,
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            },
            "offload_param": {
                "device": "cpu",
                "pin_memory": True
            },
        },
        "bf16": {
            "enabled": True
        },
        "gradient_clipping": 1.0,
        "wall_clock_breakdown": False,
    }
    
    # Сохраняем конфиг
    import json
    with open("ds_config.json", "w") as f:
        json.dump(ds_config, f, indent=2)
    print("  DeepSpeed config сохранён в ds_config.json")

    # 5. Trainer
    print("\n[5/6] Инициализация тренера...")
    
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        warmup_ratio=0.03,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        logging_steps=LOGGING_STEPS,
        eval_strategy="no",
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        optim="adamw_torch",
        weight_decay=0.001,
        lr_scheduler_type="cosine",
        seed=3407,
        bf16=True,
        gradient_checkpointing=True,
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        report_to="none",
        deepspeed="ds_config.json",  # DeepSpeed конфиг
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        args=training_args,
    )

    print_gpu_info()

    # 6. Обучение
    print("\n[6/6] Запуск обучения с DeepSpeed...")
    print("=" * 60)
    
    stats = trainer.train()
    
    print("=" * 60)
    runtime = stats.metrics['train_runtime']
    hours = int(runtime // 3600)
    mins = int((runtime % 3600) // 60)
    print(f"\nВремя обучения: {hours}ч {mins}мин ({runtime:.0f} сек)")
    print_gpu_info()

    # Сохранение
    print("\nСохранение модели...")
    model.save_pretrained("lora_qwen3_32b_deepspeed")
    tokenizer.save_pretrained("lora_qwen3_32b_deepspeed")

    print("\n" + "=" * 60)
    print("Готово!")
    print(f"  LoRA адаптер: lora_qwen3_32b_deepspeed/")
    print("=" * 60)


if __name__ == "__main__":
    main()
