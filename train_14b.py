"""
Fine-tuning Qwen3-14B с Unsloth
16-bit (bf16) LoRA на RTX 5090 (32GB) - без квантизации
"""
import torch
from functools import partial

from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

from config import TRAIN_FILE, VAL_FILE, TARGET_MODULES
from utils import print_gpu_info, format_dataset

# ============== КОНФИГУРАЦИЯ ==============
MODEL_NAME = "Qwen/Qwen3-14B"
MAX_SEQ_LENGTH = 2048

LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0

OUTPUT_DIR = "outputs_14b_16bit"
BATCH_SIZE = 1
GRAD_ACCUM = 8  # эффективный batch = 8
NUM_EPOCHS = 3
LEARNING_RATE = 2e-4
SAVE_STEPS = 500
LOGGING_STEPS = 10


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
        dtype=torch.bfloat16,
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
    dataset = load_dataset("json", data_files={"train": TRAIN_FILE, "validation": VAL_FILE})
    train_dataset = dataset["train"].map(
        partial(format_dataset, eos_token=tokenizer.eos_token), batched=True
    )

    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Effective batch: {BATCH_SIZE * GRAD_ACCUM}")

    steps_per_epoch = len(train_dataset) // (BATCH_SIZE * GRAD_ACCUM)
    print(f"  Steps/epoch: {steps_per_epoch}, Total: {steps_per_epoch * NUM_EPOCHS}")

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
    print(f"\nВремя: {int(runtime // 3600)}h {int((runtime % 3600) // 60)}min")
    print_gpu_info()

    # Save
    print("\nСохранение...")
    model.save_pretrained("lora_qwen3_14b_16bit")
    tokenizer.save_pretrained("lora_qwen3_14b_16bit")
    print("  LoRA: lora_qwen3_14b_16bit/")
    print("Готово!")


if __name__ == "__main__":
    main()
