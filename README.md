# Fine-tuning Qwen3

Fine-tuning моделей Qwen3 (8B, 14B, 32B) на казахстанском законодательстве с использованием Unsloth и LoRA.

## Структура проекта

```
├── train_8b.py          # Обучение Qwen3-8B (для GPU 16GB+)
├── train_32b.py         # Обучение Qwen3-32B (для RTX 5090 32GB)
├── train.py             # Обучение Qwen3-14B
├── train_qwen3.ipynb    # Jupyter notebook с переключением моделей
├── config.py            # Конфигурация модели и обучения
├── prepare_data.py      # Подготовка и объединение датасетов
├── inference.py         # Инференс обученной модели
├── chat.py              # Интерактивный чат с моделью
├── test_8b.py           # Тестирование модели
├── export_model.py      # Экспорт модели
├── run_training.sh      # Bash скрипт для запуска обучения
├── SETUP_RTX5090.md     # Инструкция для RTX 5090
├── combined_data/       # Объединённые датасеты
├── finetune_dataset/    # Исходный датасет для обучения
├── data_final_pars*/    # Дополнительные данные
└── datasetTranslate/    # Утилиты для перевода датасета
```

## Требования

### Для Qwen3-8B:
- Python 3.10+
- CUDA 12.0+
- GPU с 16GB+ VRAM (RTX 4080/5080 или выше)

### Для Qwen3-32B:
- Python 3.11
- CUDA 12.4+
- GPU с 32GB VRAM (RTX 5090)
- RAM: 64GB+ (рекомендуется 128GB)

### Установка зависимостей

```bash
pip install unsloth
pip install transformers datasets trl peft
```

## Быстрый старт

### Для RTX 5090 (32GB VRAM) - Qwen3-32B

См. подробную инструкцию: [SETUP_RTX5090.md](SETUP_RTX5090.md)

```bash
# Установка окружения
conda create -n ai python=3.11 -y
conda activate ai
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps xformers trl peft accelerate bitsandbytes datasets

# Запуск обучения 32B
python train_32b.py
```

### Для GPU 16GB+ VRAM - Qwen3-8B

```bash
# 1. Подготовка данных
python prepare_data.py

# 2. Запуск обучения
python train_8b.py

# Или через bash скрипт
./run_training.sh
```

### 3. Тестирование модели

```bash
python test_8b.py
```

### 4. Интерактивный чат

```bash
python chat.py
```

## Конфигурация

### Qwen3-8B (train_8b.py)

| Параметр | Значение | Описание |
|----------|----------|----------|
| MODEL_NAME | `unsloth/Qwen3-8B-unsloth-bnb-4bit` | Базовая модель |
| MAX_SEQ_LENGTH | 2048 | Максимальная длина последовательности |
| LORA_R | 16 | Ранг LoRA |
| LORA_ALPHA | 16 | Alpha LoRA |
| BATCH_SIZE | 2 | Размер батча |
| GRAD_ACCUM | 4 | Градиентная аккумуляция (эффективный batch = 8) |
| LEARNING_RATE | 2e-4 | Скорость обучения |

### Qwen3-32B (train_32b.py) - для RTX 5090

| Параметр | Значение | Описание |
|----------|----------|----------|
| MODEL_NAME | `unsloth/Qwen3-32B-unsloth-bnb-4bit` | Базовая модель |
| MAX_SEQ_LENGTH | 1024 | Уменьшено для экономии памяти |
| LORA_R | 32 | Увеличено для лучшего качества |
| LORA_ALPHA | 32 | Alpha LoRA |
| BATCH_SIZE | 1 | Минимальный для 32GB VRAM |
| GRAD_ACCUM | 8 | Градиентная аккумуляция (эффективный batch = 8) |
| LEARNING_RATE | 1e-4 | Чуть меньше для стабильности |

## Формат данных

Датасет в формате JSONL с полями:

```json
{
  "instruction": "Ваш вопрос или инструкция",
  "input": "Дополнительный контекст (опционально)",
  "output": "Ожидаемый ответ"
}
```

## Результаты

После обучения модели сохраняются в:
- `finetuned_qwen3_8b/` или `finetuned_qwen3_32b/` — полная модель с адаптерами
- `lora_qwen3_8b/` или `lora_qwen3_32b/` — только LoRA адаптеры
- `outputs_8b/` или `outputs_32b/` — чекпоинты во время обучения

## Дополнительная документация

- [SETUP_RTX5090.md](SETUP_RTX5090.md) - Подробная инструкция для RTX 5090
- [train_qwen3.ipynb](train_qwen3.ipynb) - Jupyter notebook с визуализацией и переключением моделей

## Лицензия

MIT

