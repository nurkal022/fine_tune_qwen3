"""
Fine-tuning Qwen3-8B с Unsloth
Оптимизировано для RTX 5080 (16GB VRAM)
С подробным логированием
"""
import os
import sys
import json
import torch
import logging
from datetime import datetime
from pathlib import Path

from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from transformers import TrainerCallback

# ============== ЛОГИРОВАНИЕ ==============

# Создаем папку для логов
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# Имя лог-файла с timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOG_DIR / f"train_8b_{timestamp}.log"

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ============== КОНФИГУРАЦИЯ ==============
MODEL_NAME = "unsloth/Qwen3-8B-unsloth-bnb-4bit"
MAX_SEQ_LENGTH = 2048
LOAD_IN_4BIT = True

# LoRA
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Training (оптимизировано для RTX 5080 16GB)
OUTPUT_DIR = "outputs_8b"
BATCH_SIZE = 2
GRAD_ACCUM = 4  # эффективный batch = 8
NUM_EPOCHS = 3  # увеличено для лучшего качества
LEARNING_RATE = 2e-4
SAVE_STEPS = 1000
LOGGING_STEPS = 50

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

# ============== CALLBACK ДЛЯ ЛОГИРОВАНИЯ ==============

class DetailedLoggingCallback(TrainerCallback):
    """Подробное логирование во время обучения"""
    
    def __init__(self):
        self.start_time = None
        self.step_times = []
        
    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = datetime.now()
        logger.info("=" * 60)
        logger.info("🚀 НАЧАЛО ОБУЧЕНИЯ")
        logger.info("=" * 60)
        logger.info(f"Время старта: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Всего шагов: {state.max_steps}")
        logger.info(f"Эпох: {args.num_train_epochs}")
        
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % args.logging_steps == 0:
            # Прогресс
            progress = state.global_step / state.max_steps * 100
            
            # GPU память
            if torch.cuda.is_available():
                gpu_mem = torch.cuda.max_memory_reserved() / 1024**3
                gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                gpu_percent = gpu_mem / gpu_total * 100
            else:
                gpu_mem = gpu_total = gpu_percent = 0
            
            # Оценка времени
            elapsed = (datetime.now() - self.start_time).total_seconds()
            if state.global_step > 0:
                time_per_step = elapsed / state.global_step
                remaining_steps = state.max_steps - state.global_step
                eta_seconds = remaining_steps * time_per_step
                eta_hours = int(eta_seconds // 3600)
                eta_mins = int((eta_seconds % 3600) // 60)
                eta_str = f"{eta_hours}ч {eta_mins}мин"
            else:
                eta_str = "расчёт..."
            
            logger.info(
                f"Step {state.global_step}/{state.max_steps} ({progress:.1f}%) | "
                f"Loss: {state.log_history[-1].get('loss', 'N/A') if state.log_history else 'N/A'} | "
                f"GPU: {gpu_mem:.1f}/{gpu_total:.1f}GB ({gpu_percent:.0f}%) | "
                f"ETA: {eta_str}"
            )
    
    def on_save(self, args, state, control, **kwargs):
        logger.info(f"💾 Чекпоинт сохранён: {args.output_dir}/checkpoint-{state.global_step}")
        
    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = int(state.epoch)
        logger.info(f"📊 Эпоха {epoch} завершена")
        
    def on_train_end(self, args, state, control, **kwargs):
        end_time = datetime.now()
        total_time = (end_time - self.start_time).total_seconds()
        hours = int(total_time // 3600)
        mins = int((total_time % 3600) // 60)
        secs = int(total_time % 60)
        
        logger.info("=" * 60)
        logger.info("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО")
        logger.info("=" * 60)
        logger.info(f"Время окончания: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Общее время: {hours}ч {mins}мин {secs}сек")
        logger.info(f"Всего шагов: {state.global_step}")
        if state.log_history:
            final_loss = [h.get('loss') for h in state.log_history if 'loss' in h]
            if final_loss:
                logger.info(f"Финальный loss: {final_loss[-1]:.4f}")

# ============== ФУНКЦИИ ==============

def log_gpu_info():
    """Логирование информации о GPU"""
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        reserved = round(torch.cuda.max_memory_reserved() / 1024**3, 2)
        allocated = round(torch.cuda.memory_allocated() / 1024**3, 2)
        total = round(gpu.total_memory / 1024**3, 2)
        logger.info(f"GPU: {gpu.name}")
        logger.info(f"  VRAM: {allocated}/{total} GB allocated, {reserved} GB reserved")
    else:
        logger.warning("CUDA недоступна!")

def log_config():
    """Логирование конфигурации"""
    config = {
        "model_name": MODEL_NAME,
        "max_seq_length": MAX_SEQ_LENGTH,
        "load_in_4bit": LOAD_IN_4BIT,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "batch_size": BATCH_SIZE,
        "grad_accum": GRAD_ACCUM,
        "effective_batch": BATCH_SIZE * GRAD_ACCUM,
        "num_epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "save_steps": SAVE_STEPS,
        "logging_steps": LOGGING_STEPS,
    }
    
    logger.info("📋 КОНФИГУРАЦИЯ:")
    for k, v in config.items():
        logger.info(f"  {k}: {v}")
    
    # Сохраняем конфиг в JSON
    config_file = LOG_DIR / f"config_{timestamp}.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    logger.info(f"  Конфиг сохранён: {config_file}")

def main():
    logger.info("=" * 60)
    logger.info("Fine-tuning Qwen3-8B")
    logger.info(f"Лог-файл: {LOG_FILE}")
    logger.info("=" * 60)
    
    log_config()
    log_gpu_info()
    
    # 1. Загрузка модели
    logger.info("\n[1/5] Загрузка модели...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=LOAD_IN_4BIT,
    )
    logger.info("  ✅ Модель загружена")
    log_gpu_info()
    
    # 2. LoRA
    logger.info("\n[2/5] Применение LoRA...")
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
    
    # Подсчет параметров
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"  Trainable params: {trainable:,} ({trainable/total*100:.2f}%)")
    logger.info(f"  Total params: {total:,}")
    logger.info("  ✅ LoRA применен")
    
    # 3. Датасет
    logger.info("\n[3/5] Загрузка датасета...")
    EOS_TOKEN = tokenizer.eos_token
    
    def formatting_func(examples):
        texts = []
        for instr, inp, out in zip(examples["instruction"], examples["input"], examples["output"]):
            texts.append(ALPACA_PROMPT.format(instr, inp, out) + EOS_TOKEN)
        return {"text": texts}
    
    dataset = load_dataset("json", data_files={"train": TRAIN_FILE, "validation": VAL_FILE})
    train_dataset = dataset["train"].map(formatting_func, batched=True)
    
    logger.info(f"  Train samples: {len(train_dataset):,}")
    logger.info(f"  Validation samples: {len(dataset['validation']):,}")
    
    # Расчёт шагов
    effective_batch = BATCH_SIZE * GRAD_ACCUM
    steps_per_epoch = len(train_dataset) // effective_batch
    total_steps = steps_per_epoch * NUM_EPOCHS
    logger.info(f"  Steps per epoch: {steps_per_epoch:,}")
    logger.info(f"  Total steps: ~{total_steps:,}")
    logger.info("  ✅ Датасет загружен")
    
    # 4. Trainer
    logger.info("\n[4/5] Инициализация тренера...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        callbacks=[DetailedLoggingCallback()],
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
    logger.info("  ✅ Trainer создан")
    log_gpu_info()
    
    # 5. Train
    logger.info("\n[5/5] Запуск обучения...")
    stats = trainer.train()
    
    # Результаты
    runtime = stats.metrics['train_runtime']
    hours = int(runtime // 3600)
    mins = int((runtime % 3600) // 60)
    
    logger.info(f"\n📊 РЕЗУЛЬТАТЫ:")
    logger.info(f"  Время обучения: {hours}ч {mins}мин ({runtime:.0f} сек)")
    logger.info(f"  Train loss: {stats.metrics.get('train_loss', 'N/A')}")
    logger.info(f"  Train samples/sec: {stats.metrics.get('train_samples_per_second', 'N/A'):.2f}")
    log_gpu_info()
    
    # Save
    logger.info("\n💾 Сохранение модели...")
    model.save_pretrained("finetuned_qwen3_8b")
    tokenizer.save_pretrained("finetuned_qwen3_8b")
    logger.info("  ✅ Сохранено: finetuned_qwen3_8b/")
    
    model.save_pretrained("lora_qwen3_8b")
    tokenizer.save_pretrained("lora_qwen3_8b")
    logger.info("  ✅ Сохранено: lora_qwen3_8b/")
    
    # Сохраняем метрики
    metrics_file = LOG_DIR / f"metrics_{timestamp}.json"
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(stats.metrics, f, indent=2, ensure_ascii=False)
    logger.info(f"  ✅ Метрики сохранены: {metrics_file}")
    
    logger.info("\n" + "=" * 60)
    logger.info("🎉 ГОТОВО!")
    logger.info("=" * 60)
    logger.info(f"  Модель: finetuned_qwen3_8b/")
    logger.info(f"  LoRA: lora_qwen3_8b/")
    logger.info(f"  Логи: {LOG_FILE}")

if __name__ == "__main__":
    main()
