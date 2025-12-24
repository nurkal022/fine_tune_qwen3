"""
Fine-tuning Qwen3-8B с Unsloth
Отдельный скрипт для 8B модели
"""
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

# ============== КОНФИГУРАЦИЯ ==============
MODEL_NAME = "unsloth/Qwen3-8B-unsloth-bnb-4bit"
MAX_SEQ_LENGTH = 2048
LOAD_IN_4BIT = True

# LoRA
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Training
OUTPUT_DIR = "outputs_8b"
BATCH_SIZE = 2
GRAD_ACCUM = 4  # эффективный batch = 8
NUM_EPOCHS = 1
LEARNING_RATE = 2e-4
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

# ============== ФУНКЦИИ ==============

def print_gpu_info():
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        reserved = round(torch.cuda.max_memory_reserved() / 1024**3, 2)
        total = round(gpu.total_memory / 1024**3, 2)
        print(f"GPU: {gpu.name} | {reserved}/{total} GB used")

def main():
    print("=" * 50)
    print("Fine-tuning Qwen3-8B")
    print("=" * 50)
    print_gpu_info()
    
    # 1. Загрузка модели
    print("\n[1/5] Загрузка модели...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=LOAD_IN_4BIT,
    )
    
    # 2. LoRA
    print("\n[2/5] Применение LoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    
    # 3. Датасет
    print("\n[3/5] Загрузка датасета...")
    EOS_TOKEN = tokenizer.eos_token
    
    def formatting_func(examples):
        texts = []
        for instr, inp, out in zip(examples["instruction"], examples["input"], examples["output"]):
            texts.append(ALPACA_PROMPT.format(instr, inp, out) + EOS_TOKEN)
        return {"text": texts}
    
    dataset = load_dataset("json", data_files={"train": TRAIN_FILE, "validation": VAL_FILE})
    train_dataset = dataset["train"].map(formatting_func, batched=True)
    
    print(f"  Train: {len(train_dataset)} samples")
    
    # 4. Trainer
    print("\n[4/5] Инициализация тренера...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        args=SFTConfig(
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
            optim="adamw_8bit",
            weight_decay=0.001,
            lr_scheduler_type="linear",
            seed=3407,
            report_to="none",
            bf16=True,
        ),
    )
    
    print_gpu_info()
    
    # 5. Train
    print("\n[5/5] Запуск обучения...")
    print("=" * 50)
    stats = trainer.train()
    print("=" * 50)
    
    runtime = stats.metrics['train_runtime']
    print(f"\nВремя: {runtime:.0f} сек ({runtime/60:.1f} мин)")
    print_gpu_info()
    
    # Save
    print("\nСохранение модели...")
    model.save_pretrained("finetuned_qwen3_8b")
    tokenizer.save_pretrained("finetuned_qwen3_8b")
    
    # LoRA only
    model.save_pretrained("lora_qwen3_8b")
    tokenizer.save_pretrained("lora_qwen3_8b")
    
    print("\nГотово!")
    print(f"  Модель: finetuned_qwen3_8b/")
    print(f"  LoRA: lora_qwen3_8b/")

if __name__ == "__main__":
    main()

