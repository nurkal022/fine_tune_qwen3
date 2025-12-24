# Fine-tuning Qwen3-8B

Fine-tuning Qwen3-8B модели на казахстанском законодательстве с использованием Unsloth и LoRA.

## Структура проекта

```
├── train_8b.py          # Основной скрипт обучения Qwen3-8B
├── train.py             # Скрипт обучения Qwen3-14B
├── config.py            # Конфигурация модели и обучения
├── prepare_data.py      # Подготовка и объединение датасетов
├── inference.py         # Инференс обученной модели
├── chat.py              # Интерактивный чат с моделью
├── test_8b.py           # Тестирование модели
├── export_model.py      # Экспорт модели
├── run_training.sh      # Bash скрипт для запуска обучения
├── combined_data/       # Объединённые датасеты
├── finetune_dataset/    # Исходный датасет для обучения
├── data_final_pars*/    # Дополнительные данные
└── datasetTranslate/    # Утилиты для перевода датасета
```

## Требования

- Python 3.10+
- CUDA 12.0+
- GPU с 16GB+ VRAM (RTX 4080/5080 или выше)

### Установка зависимостей

```bash
pip install unsloth
pip install transformers datasets trl peft
```

## Использование

### 1. Подготовка данных

```bash
python prepare_data.py
```

### 2. Запуск обучения

```bash
python train_8b.py
```

Или через bash скрипт:

```bash
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

Основные параметры в `train_8b.py`:

| Параметр | Значение | Описание |
|----------|----------|----------|
| MODEL_NAME | `unsloth/Qwen3-8B-unsloth-bnb-4bit` | Базовая модель |
| MAX_SEQ_LENGTH | 2048 | Максимальная длина последовательности |
| LORA_R | 16 | Ранг LoRA |
| LORA_ALPHA | 16 | Alpha LoRA |
| BATCH_SIZE | 2 | Размер батча |
| GRAD_ACCUM | 4 | Градиентная аккумуляция |
| LEARNING_RATE | 2e-4 | Скорость обучения |

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

После обучения модель сохраняется в:
- `finetuned_qwen3_8b/` — полная модель с адаптерами
- `lora_qwen3_8b/` — только LoRA адаптеры

## Лицензия

MIT

