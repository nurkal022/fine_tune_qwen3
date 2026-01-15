"""
Fine-tuning Qwen3-14B с Unsloth
16-bit (bf16) LoRA на RTX 5090 (32GB) - максимальное качество, без квантизации
"""
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

# ============== КОНФИГУРАЦИЯ ==============
MODEL_NAME = "Qwen/Qwen3-14B"  # Базовая модель
MAX_SEQ_LENGTH = 2048
# 16-bit (bf16) — без квантизации, максимальное качество
# Займёт ~28-30GB VRAM

# LoRA
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Training
OUTPUT_DIR = "outputs_14b_16bit"  # 16-bit версия
BATCH_SIZE = 1  # уменьшено для 16-bit (больше памяти)
GRAD_ACCUM = 8  # эффективный batch = 8
NUM_EPOCHS = 3  # увеличено для лучшего обучения
LEARNING_RATE = 2e-4
SAVE_STEPS = 500
LOGGING_STEPS = 10

# Расчёт времени (примерно)
# 16-bit медленнее 4-bit, но качество лучше
# Ожидаемое время: ~12-15 часов на 3 эпохи

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
        for i in range(torch.cuda.device_count()):
            gpu = torch.cuda.get_device_properties(i)
            reserved = round(torch.cuda.memory_reserved(i) / 1024**3, 2)
            allocated = round(torch.cuda.memory_allocated(i) / 1024**3, 2)
            total = round(gpu.total_memory / 1024**3, 2)
            print(f"GPU {i}: {gpu.name} | {allocated}/{total} GB allocated")

def main():
    print("=" * 60)
    print("Fine-tuning Qwen3-14B (16-bit bf16 LoRA)")
    print("=" * 60)
    print_gpu_info()
    
    # 1. Загрузка модели
    print("\n[1/5] Загрузка модели Qwen3-14B (16-bit bf16)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.bfloat16,  # 16-bit без квантизации
        load_in_4bit=False,
    )
    
    print_gpu_info()
    
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
    print(f"  Effective batch size: {BATCH_SIZE * GRAD_ACCUM}")
    
    # Расчёт времени обучения
    steps_per_epoch = len(train_dataset) // (BATCH_SIZE * GRAD_ACCUM)
    total_steps = steps_per_epoch * NUM_EPOCHS
    estimated_time_per_step = 1.5  # секунд (примерно)
    estimated_total_time = total_steps * estimated_time_per_step
    estimated_hours = int(estimated_total_time // 3600)
    estimated_mins = int((estimated_total_time % 3600) // 60)
    
    print(f"\n  📊 Прогноз обучения:")
    print(f"     Эпох: {NUM_EPOCHS}")
    print(f"     Шагов на эпоху: {steps_per_epoch}")
    print(f"     Всего шагов: {total_steps}")
    print(f"     Примерное время: ~{estimated_hours}ч {estimated_mins}мин")
    
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
    print("=" * 60)
    stats = trainer.train()
    print("=" * 60)
    
    runtime = stats.metrics['train_runtime']
    hours = int(runtime // 3600)
    mins = int((runtime % 3600) // 60)
    print(f"\nВремя: {hours}ч {mins}мин ({runtime:.0f} сек)")
    print_gpu_info()
    
    # Save
    print("\nСохранение модели...")
    model.save_pretrained("finetuned_qwen3_14b_16bit")
    tokenizer.save_pretrained("finetuned_qwen3_14b_16bit")
    
    # LoRA only
    model.save_pretrained("lora_qwen3_14b_16bit")
    tokenizer.save_pretrained("lora_qwen3_14b_16bit")
    
    print("\nГотово!")
    print(f"  Модель: finetuned_qwen3_14b_16bit/")
    print(f"  LoRA: lora_qwen3_14b_16bit/")

if __name__ == "__main__":
    main()
