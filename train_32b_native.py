"""
Fine-tuning Qwen3-32B - Native (без Unsloth)
Для RTX 5090 (Blackwell) + PyTorch Nightly
4-bit QLoRA с bitsandbytes
"""
import torch
from functools import partial

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

from config import TRAIN_FILE, VAL_FILE, TARGET_MODULES, ALPACA_PROMPT
from utils import print_gpu_info

# ============== КОНФИГУРАЦИЯ ==============
MODEL_NAME = "Qwen/Qwen2.5-32B"
MAX_SEQ_LENGTH = 2048
OUTPUT_DIR = "outputs_32b_native"

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

BATCH_SIZE = 1
GRAD_ACCUM = 8  # эффективный batch = 8
NUM_EPOCHS = 1
LEARNING_RATE = 1e-4
SAVE_STEPS = 500
LOGGING_STEPS = 10


def main():
    print("=" * 60)
    print("Fine-tuning Qwen3-32B (Native - без Unsloth)")
    print("4-bit QLoRA на RTX 5090")
    print("=" * 60)
    print_gpu_info()

    # 1. Квантизация конфиг
    print("\n[1/6] Настройка 4-bit квантизации...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # 2. Загрузка модели
    print("\n[2/6] Загрузка модели...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print_gpu_info()

    # 3. QLoRA
    print("\n[3/6] Подготовка QLoRA...")
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 4. Датасет
    print("\n[4/6] Загрузка датасета...")

    def formatting_func(examples):
        texts = []
        for instr, inp, out in zip(examples["instruction"], examples["input"], examples["output"]):
            texts.append(ALPACA_PROMPT.format(instr, inp, out) + tokenizer.eos_token)
        return {"text": texts}

    dataset = load_dataset("json", data_files={"train": TRAIN_FILE, "validation": VAL_FILE})
    train_dataset = dataset["train"].map(formatting_func, batched=True, remove_columns=dataset["train"].column_names)

    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Effective batch: {BATCH_SIZE * GRAD_ACCUM}")

    # 5. Trainer
    print("\n[5/6] Инициализация тренера...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
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
            optim="paged_adamw_8bit",
            weight_decay=0.001,
            lr_scheduler_type="cosine",
            seed=3407,
            bf16=True,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            max_seq_length=MAX_SEQ_LENGTH,
            dataset_text_field="text",
            report_to="none",
        ),
    )
    print_gpu_info()

    # 6. Train
    print("\n[6/6] Запуск обучения...")
    print("=" * 60)
    stats = trainer.train()
    print("=" * 60)

    runtime = stats.metrics['train_runtime']
    print(f"\nВремя: {int(runtime // 3600)}h {int((runtime % 3600) // 60)}min")
    print_gpu_info()

    # Save
    print("\nСохранение...")
    model.save_pretrained("lora_qwen3_32b_native")
    tokenizer.save_pretrained("lora_qwen3_32b_native")
    print("  LoRA: lora_qwen3_32b_native/")
    print("Готово!")


if __name__ == "__main__":
    main()
