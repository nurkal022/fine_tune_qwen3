"""
Fine-tuning Qwen3-32B с Unsloth для RTX 5090 (32GB VRAM)
Оптимизировано для работы на одной видеокарте с 32GB памяти
"""
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

# ============== КОНФИГУРАЦИЯ для 32B на 32GB VRAM ==============
MODEL_NAME = "unsloth/Qwen3-32B-unsloth-bnb-4bit"
MAX_SEQ_LENGTH = 1024  # Уменьшено для экономии памяти
LOAD_IN_4BIT = True

# LoRA (увеличиваем для 32B)
LORA_R = 32  # Увеличено для лучшего качества
LORA_ALPHA = 32
LORA_DROPOUT = 0
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Training (консервативные настройки для 32GB VRAM)
OUTPUT_DIR = "outputs_32b"
BATCH_SIZE = 1  # Минимальный для экономии памяти
GRAD_ACCUM = 8  # Эффективный batch = 8
NUM_EPOCHS = 1
LEARNING_RATE = 1e-4  # Чуть меньше для стабильности
SAVE_STEPS = 200
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
        allocated = round(torch.cuda.memory_allocated() / 1024**3, 2)
        total = round(gpu.total_memory / 1024**3, 2)
        free = total - reserved
        print(f"GPU: {gpu.name} | Reserved: {reserved} GB | Allocated: {allocated} GB | Free: {free} GB | Total: {total} GB")
    else:
        print("⚠️  CUDA недоступна!")

def main():
    print("=" * 60)
    print("Fine-tuning Qwen3-32B on RTX 5090 (32GB VRAM)")
    print("=" * 60)
    print_gpu_info()
    
    # 1. Загрузка модели
    print("\n[1/5] Загрузка модели 32B...")
    print(f"   Model: {MODEL_NAME}")
    print(f"   Max seq length: {MAX_SEQ_LENGTH}")
    print(f"   4-bit quantization: {LOAD_IN_4BIT}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=LOAD_IN_4BIT,
    )
    
    print_gpu_info()
    
    # 2. LoRA
    print("\n[2/5] Применение LoRA...")
    print(f"   r={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}")
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",  # Критически важно для 32B!
        random_state=3407,
    )
    
    # Подсчет параметров
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_percent = trainable_params / total_params * 100
    print(f"   Trainable params: {trainable_params:,} ({trainable_percent:.2f}%)")
    print(f"   Total params: {total_params:,}")
    
    print_gpu_info()
    
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
    
    print(f"  Train: {len(train_dataset):,} samples")
    
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
            gradient_checkpointing=True,  # Дополнительная экономия памяти
        ),
    )
    
    effective_batch = BATCH_SIZE * GRAD_ACCUM
    print(f"   Effective batch size: {BATCH_SIZE} × {GRAD_ACCUM} = {effective_batch}")
    print(f"   Learning rate: {LEARNING_RATE}")
    print(f"   Save steps: {SAVE_STEPS}")
    
    print_gpu_info()
    
    # 5. Train
    print("\n[5/5] Запуск обучения...")
    print("=" * 60)
    print("⚠️  ВНИМАНИЕ: Обучение 32B модели может занять много времени!")
    print("=" * 60)
    
    stats = trainer.train()
    
    print("=" * 60)
    runtime = stats.metrics['train_runtime']
    print(f"\n✅ Обучение завершено!")
    print(f"   Время: {runtime:.0f} сек ({runtime/60:.1f} мин)")
    print(f"   Final loss: {stats.metrics.get('train_loss', 'N/A')}")
    print_gpu_info()
    
    # Save
    print("\n💾 Сохранение модели...")
    model.save_pretrained("finetuned_qwen3_32b")
    tokenizer.save_pretrained("finetuned_qwen3_32b")
    
    model.save_pretrained("lora_qwen3_32b")
    tokenizer.save_pretrained("lora_qwen3_32b")
    
    print("\n✅ Готово!")
    print(f"   Полная модель: finetuned_qwen3_32b/")
    print(f"   LoRA адаптеры: lora_qwen3_32b/")

if __name__ == "__main__":
    main()

