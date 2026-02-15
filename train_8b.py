"""
Fine-tuning Qwen3-8B с Unsloth
Оптимизировано для RTX 5080 (16GB VRAM)
"""
import sys
import json
import torch
import logging
from datetime import datetime
from pathlib import Path
from functools import partial

from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from transformers import TrainerCallback

from config import TRAIN_FILE, VAL_FILE, TARGET_MODULES
from utils import log_gpu_info, format_dataset

# ============== ЛОГИРОВАНИЕ ==============

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOG_DIR / f"train_8b_{timestamp}.log"

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

LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0

OUTPUT_DIR = "outputs_8b"
BATCH_SIZE = 2
GRAD_ACCUM = 4  # эффективный batch = 8
NUM_EPOCHS = 3
LEARNING_RATE = 2e-4
SAVE_STEPS = 1000
LOGGING_STEPS = 50


# ============== CALLBACK ДЛЯ ЛОГИРОВАНИЯ ==============

class DetailedLoggingCallback(TrainerCallback):
    def __init__(self):
        self.start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = datetime.now()
        logger.info("=" * 60)
        logger.info("НАЧАЛО ОБУЧЕНИЯ")
        logger.info(f"Всего шагов: {state.max_steps}, Эпох: {args.num_train_epochs}")
        logger.info("=" * 60)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % args.logging_steps == 0:
            progress = state.global_step / state.max_steps * 100
            gpu_mem = gpu_total = gpu_percent = 0
            if torch.cuda.is_available():
                gpu_mem = torch.cuda.max_memory_reserved() / 1024**3
                gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                gpu_percent = gpu_mem / gpu_total * 100

            elapsed = (datetime.now() - self.start_time).total_seconds()
            if state.global_step > 0:
                remaining = (state.max_steps - state.global_step) * (elapsed / state.global_step)
                eta_str = f"{int(remaining // 3600)}h {int((remaining % 3600) // 60)}min"
            else:
                eta_str = "..."

            loss = state.log_history[-1].get('loss', 'N/A') if state.log_history else 'N/A'
            logger.info(
                f"Step {state.global_step}/{state.max_steps} ({progress:.1f}%) | "
                f"Loss: {loss} | GPU: {gpu_mem:.1f}/{gpu_total:.1f}GB ({gpu_percent:.0f}%) | "
                f"ETA: {eta_str}"
            )

    def on_save(self, args, state, control, **kwargs):
        logger.info(f"Checkpoint: {args.output_dir}/checkpoint-{state.global_step}")

    def on_epoch_end(self, args, state, control, **kwargs):
        logger.info(f"Epoch {int(state.epoch)} complete")

    def on_train_end(self, args, state, control, **kwargs):
        total = (datetime.now() - self.start_time).total_seconds()
        logger.info("=" * 60)
        logger.info(f"ОБУЧЕНИЕ ЗАВЕРШЕНО за {int(total // 3600)}h {int((total % 3600) // 60)}min")
        if state.log_history:
            final_loss = [h.get('loss') for h in state.log_history if 'loss' in h]
            if final_loss:
                logger.info(f"Финальный loss: {final_loss[-1]:.4f}")
        logger.info("=" * 60)


# ============== MAIN ==============

def log_config():
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
    }
    logger.info("КОНФИГУРАЦИЯ:")
    for k, v in config.items():
        logger.info(f"  {k}: {v}")

    config_file = LOG_DIR / f"config_{timestamp}.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def main():
    logger.info("=" * 60)
    logger.info("Fine-tuning Qwen3-8B")
    logger.info(f"Log: {LOG_FILE}")
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
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Trainable: {trainable:,} ({trainable/total_params*100:.2f}%)")

    # 3. Датасет
    logger.info("\n[3/5] Загрузка датасета...")
    dataset = load_dataset("json", data_files={"train": TRAIN_FILE, "validation": VAL_FILE})
    train_dataset = dataset["train"].map(
        partial(format_dataset, eos_token=tokenizer.eos_token), batched=True
    )
    logger.info(f"  Train: {len(train_dataset):,}, Val: {len(dataset['validation']):,}")

    effective_batch = BATCH_SIZE * GRAD_ACCUM
    steps_per_epoch = len(train_dataset) // effective_batch
    logger.info(f"  Steps/epoch: {steps_per_epoch:,}, Total: ~{steps_per_epoch * NUM_EPOCHS:,}")

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
    log_gpu_info()

    # 5. Train
    logger.info("\n[5/5] Запуск обучения...")
    stats = trainer.train()

    runtime = stats.metrics['train_runtime']
    logger.info(f"\nВремя: {int(runtime // 3600)}h {int((runtime % 3600) // 60)}min")
    logger.info(f"Train loss: {stats.metrics.get('train_loss', 'N/A')}")
    log_gpu_info()

    # Save
    logger.info("\nСохранение...")
    model.save_pretrained("lora_qwen3_8b")
    tokenizer.save_pretrained("lora_qwen3_8b")
    logger.info("  lora_qwen3_8b/")

    metrics_file = LOG_DIR / f"metrics_{timestamp}.json"
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(stats.metrics, f, indent=2, ensure_ascii=False)

    logger.info("\nГОТОВО!")


if __name__ == "__main__":
    main()
