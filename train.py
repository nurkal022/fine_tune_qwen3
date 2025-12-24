"""
Основной скрипт обучения Qwen3-14B с Unsloth
"""
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

from config import (
    ModelConfig, LoraConfig, TrainingConfig, DataConfig, ALPACA_PROMPT
)

def print_gpu_info():
    """Вывод информации о GPU"""
    if torch.cuda.is_available():
        gpu_stats = torch.cuda.get_device_properties(0)
        memory_reserved = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"GPU: {gpu_stats.name}")
        print(f"Max memory: {max_memory} GB")
        print(f"Reserved: {memory_reserved} GB")
    else:
        print("CUDA not available!")

def load_model(model_cfg: ModelConfig):
    """Загрузка модели"""
    print("\n[1/5] Загрузка модели...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg.model_name,
        max_seq_length=model_cfg.max_seq_length,
        dtype=model_cfg.dtype,
        load_in_4bit=model_cfg.load_in_4bit,
    )
    return model, tokenizer

def apply_lora(model, lora_cfg: LoraConfig):
    """Применение LoRA адаптеров"""
    print("\n[2/5] Применение LoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg.r,
        target_modules=list(lora_cfg.target_modules),
        lora_alpha=lora_cfg.lora_alpha,
        lora_dropout=lora_cfg.lora_dropout,
        bias=lora_cfg.bias,
        use_gradient_checkpointing=lora_cfg.use_gradient_checkpointing,
        random_state=lora_cfg.random_state,
        use_rslora=lora_cfg.use_rslora,
        loftq_config=None,
    )
    return model

def prepare_dataset(tokenizer, data_cfg: DataConfig):
    """Подготовка датасета"""
    print("\n[3/5] Загрузка датасета...")
    
    EOS_TOKEN = tokenizer.eos_token
    
    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]
        texts = []
        for instruction, input_text, output in zip(instructions, inputs, outputs):
            text = ALPACA_PROMPT.format(instruction, input_text, output) + EOS_TOKEN
            texts.append(text)
        return {"text": texts}
    
    dataset = load_dataset("json", data_files={
        "train": data_cfg.combined_train,
        "validation": data_cfg.combined_val
    })
    
    train_dataset = dataset["train"].map(formatting_prompts_func, batched=True)
    eval_dataset = dataset["validation"].map(formatting_prompts_func, batched=True)
    
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Eval samples: {len(eval_dataset)}")
    
    return train_dataset, eval_dataset

def create_trainer(model, tokenizer, train_dataset, eval_dataset, 
                   model_cfg: ModelConfig, train_cfg: TrainingConfig):
    """Создание тренера"""
    print("\n[4/5] Инициализация тренера...")
    
    # Если eval отключен, не передаём eval_dataset для экономии памяти
    use_eval = train_cfg.eval_strategy != "no"
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if use_eval else None,
        dataset_text_field="text",
        max_seq_length=model_cfg.max_seq_length,
        args=SFTConfig(
            output_dir=train_cfg.output_dir,
            per_device_train_batch_size=train_cfg.per_device_train_batch_size,
            gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
            warmup_ratio=train_cfg.warmup_ratio,
            num_train_epochs=train_cfg.num_train_epochs,
            learning_rate=train_cfg.learning_rate,
            logging_steps=train_cfg.logging_steps,
            eval_strategy=train_cfg.eval_strategy,
            eval_steps=train_cfg.eval_steps,
            save_strategy=train_cfg.save_strategy,
            save_steps=train_cfg.save_steps,
            optim=train_cfg.optim,
            weight_decay=train_cfg.weight_decay,
            lr_scheduler_type=train_cfg.lr_scheduler_type,
            seed=train_cfg.seed,
            report_to=train_cfg.report_to,
            fp16=train_cfg.fp16,
            bf16=train_cfg.bf16,
        ),
    )
    return trainer

def train(trainer):
    """Запуск обучения"""
    print("\n[5/5] Запуск обучения...")
    print("=" * 50)
    
    trainer_stats = trainer.train()
    
    print("=" * 50)
    print("\nОбучение завершено!")
    
    # Статистика
    runtime = trainer_stats.metrics['train_runtime']
    print(f"Время обучения: {runtime:.2f} сек ({runtime/60:.2f} мин)")
    
    if torch.cuda.is_available():
        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        print(f"Пиковое использование памяти: {used_memory} GB")
    
    return trainer_stats

def save_model(model, tokenizer, output_path: str = "finetuned_qwen3_14b"):
    """Сохранение модели"""
    print(f"\nСохранение модели в {output_path}...")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print("Модель сохранена!")

def main():
    print("=" * 50)
    print("Fine-tuning Qwen3-14B на объединенном датасете")
    print("=" * 50)
    
    # Конфигурации
    model_cfg = ModelConfig()
    lora_cfg = LoraConfig()
    train_cfg = TrainingConfig()
    data_cfg = DataConfig()
    
    print("\nИнформация о GPU:")
    print_gpu_info()
    
    # 1. Загрузка модели
    model, tokenizer = load_model(model_cfg)
    
    # 2. Применение LoRA
    model = apply_lora(model, lora_cfg)
    
    # 3. Подготовка датасета
    train_dataset, eval_dataset = prepare_dataset(tokenizer, data_cfg)
    
    # 4. Создание тренера
    trainer = create_trainer(
        model, tokenizer, train_dataset, eval_dataset,
        model_cfg, train_cfg
    )
    
    print("\nИнформация о памяти перед обучением:")
    print_gpu_info()
    
    # 5. Обучение
    trainer_stats = train(trainer)
    
    # 6. Сохранение
    save_model(model, tokenizer, "finetuned_qwen3_14b")
    
    # Дополнительно сохраняем только LoRA адаптеры
    save_model(model, tokenizer, "lora_qwen3_14b")
    
    print("\n" + "=" * 50)
    print("Всё готово!")
    print("=" * 50)

if __name__ == "__main__":
    main()

