"""
Fine-tuning Qwen3-32B с DeepSpeed ZeRO-3
Распределение модели между 2x RTX 5090 (64GB суммарно)
БЕЗ квантизации - используем BF16
"""
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

from config import TRAIN_FILE, VAL_FILE, TARGET_MODULES, ALPACA_PROMPT
from utils import print_gpu_info

# ============== КОНФИГУРАЦИЯ ==============
MODEL_NAME = "Qwen/Qwen2.5-32B"
MAX_SEQ_LENGTH = 2048
OUTPUT_DIR = "outputs_32b_deepspeed"

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

BATCH_SIZE = 1
GRAD_ACCUM = 8  # эффективный batch = 8 * 2 GPU = 16
NUM_EPOCHS = 1
LEARNING_RATE = 1e-4
SAVE_STEPS = 500
LOGGING_STEPS = 10


def main():
    print("=" * 60)
    print("Fine-tuning Qwen3-32B с DeepSpeed ZeRO-3")
    print("Распределение между 2x RTX 5090 (64GB)")
    print("=" * 60)
    print_gpu_info()

    # 1. Загрузка модели
    print("\n[1/6] Загрузка модели (BF16)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        dtype=torch.bfloat16,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
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
            texts.append(ALPACA_PROMPT.format(instr, inp, out) + tokenizer.eos_token)
        return {"text": texts}

    dataset = load_dataset("json", data_files={"train": TRAIN_FILE, "validation": VAL_FILE})
    train_dataset = dataset["train"].map(formatting_func, batched=True, remove_columns=dataset["train"].column_names)

    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Effective batch: {BATCH_SIZE * GRAD_ACCUM * torch.cuda.device_count()}")

    # 4. DeepSpeed конфиг
    print("\n[4/6] Настройка DeepSpeed ZeRO-3...")
    ds_config = {
        "train_batch_size": BATCH_SIZE * GRAD_ACCUM * torch.cuda.device_count(),
        "train_micro_batch_size_per_gpu": BATCH_SIZE,
        "gradient_accumulation_steps": GRAD_ACCUM,
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {"device": "cpu", "pin_memory": True},
            "offload_param": {"device": "cpu", "pin_memory": True},
        },
        "bf16": {"enabled": True},
        "gradient_clipping": 1.0,
        "wall_clock_breakdown": False,
    }

    with open("ds_config.json", "w") as f:
        json.dump(ds_config, f, indent=2)

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
            optim="adamw_torch",
            weight_decay=0.001,
            lr_scheduler_type="cosine",
            seed=3407,
            bf16=True,
            gradient_checkpointing=True,
            max_seq_length=MAX_SEQ_LENGTH,
            dataset_text_field="text",
            report_to="none",
            deepspeed="ds_config.json",
        ),
    )
    print_gpu_info()

    # 6. Train
    print("\n[6/6] Запуск обучения с DeepSpeed...")
    print("=" * 60)
    stats = trainer.train()
    print("=" * 60)

    runtime = stats.metrics['train_runtime']
    print(f"\nВремя: {int(runtime // 3600)}h {int((runtime % 3600) // 60)}min")
    print_gpu_info()

    # Save
    print("\nСохранение...")
    model.save_pretrained("lora_qwen3_32b_deepspeed")
    tokenizer.save_pretrained("lora_qwen3_32b_deepspeed")
    print("  LoRA: lora_qwen3_32b_deepspeed/")
    print("Готово!")


if __name__ == "__main__":
    main()
