# Fine-tuning Qwen3-14B для казахстанского законодательства

Fine-tuned Qwen3-14B модель на ~36,000 примерах казахстанского законодательства с использованием Unsloth + LoRA.

## Результаты обучения

| Параметр | Значение |
|----------|----------|
| Базовая модель | `Qwen/Qwen3-14B` |
| Метод | LoRA (16-bit bf16) |
| Датасет | 32,440 train / 3,606 validation |
| Эпохи | 3 |
| Шаги | 21,303 |
| GPU | RTX 5090 (32GB) |
| Время обучения | ~15 часов |

## Структура проекта

```
├── train_32b.py          # Основной скрипт обучения Qwen3-14B (16-bit)
├── train_8b.py           # Скрипт обучения Qwen3-8B (4-bit)
├── test_14b_chat.py      # Интерактивный чат с обученной моделью
├── benchy.py             # Бенчмарк модели
├── prepare_data.py       # Подготовка датасетов
├── requirements.txt      # Зависимости
├── finetune_dataset/     # Основной датасет
│   ├── train.jsonl       # 32,440 примеров
│   └── validation.jsonl  # 3,606 примеров
└── combined_data/        # Объединённые данные
```

## Требования

- Python 3.10+
- CUDA 12.0+
- GPU с 24GB+ VRAM (RTX 4090/5090)

## Установка

```bash
# Создание виртуального окружения
python -m venv venv
source venv/bin/activate

# Установка зависимостей
pip install unsloth
pip install -r requirements.txt
```

## Использование

### Обучение модели

```bash
# 14B модель (16-bit, ~30GB VRAM)
CUDA_VISIBLE_DEVICES=0 python train_32b.py

# 8B модель (4-bit, ~12GB VRAM)
CUDA_VISIBLE_DEVICES=0 python train_8b.py
```

### Тестирование обученной модели

```bash
python test_14b_chat.py
```

Команды в чате:
- `exit` — выход
- `test` — запустить тестовые примеры
- `temp 0.5` — изменить temperature

### Бенчмарк

```bash
python benchy.py
```

## Конфигурация обучения

| Параметр | 14B (16-bit) | 8B (4-bit) |
|----------|--------------|------------|
| MAX_SEQ_LENGTH | 2048 | 2048 |
| LORA_R | 16 | 16 |
| LORA_ALPHA | 16 | 16 |
| BATCH_SIZE | 1 | 2 |
| GRAD_ACCUM | 8 | 4 |
| LEARNING_RATE | 2e-4 | 2e-4 |
| NUM_EPOCHS | 3 | 3 |

## Формат данных

JSONL с Alpaca форматом:

```json
{
  "instruction": "Ответь на вопрос по казахстанскому законодательству.",
  "input": "Можно ли заключить договор ГПХ с указанием цены работы за сутки?",
  "output": "Согласно законодательству РК..."
}
```

## Обученные модели

После обучения модели сохраняются в:
- `outputs_14b_16bit/checkpoint-21303/` — чекпоинты
- `finetuned_qwen3_14b_16bit/` — финальная модель
- `lora_qwen3_14b_16bit/` — только LoRA адаптеры

### Загрузка на HuggingFace

```bash
# Установка huggingface-cli
pip install huggingface_hub

# Логин
huggingface-cli login

# Загрузка LoRA адаптеров
huggingface-cli upload YOUR_USERNAME/qwen3-14b-kz-law-lora ./lora_qwen3_14b_16bit
```

## Пример использования

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="outputs_14b_16bit/checkpoint-21303",
    max_seq_length=2048,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

prompt = """Below is an instruction that describes a task, paired with an input that provides further context.

### Instruction:
Ответь на вопрос по казахстанскому законодательству.

### Input:
Какие права имеют работники при увольнении?

### Response:
"""

inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Лицензия

MIT
