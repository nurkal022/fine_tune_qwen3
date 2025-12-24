"""
Конфигурация для перевода датасета
"""
from pathlib import Path

# Пути
BASE_DIR = Path(__file__).parent.parent
INPUT_DIR = BASE_DIR / "combined_data"
OUTPUT_DIR = Path(__file__).parent / "output"

# Входные файлы
TRAIN_FILE = INPUT_DIR / "train.jsonl"
VAL_FILE = INPUT_DIR / "validation.jsonl"

# Выходные файлы
TRAIN_TRANSLATED = OUTPUT_DIR / "train_translated.jsonl"
VAL_TRANSLATED = OUTPUT_DIR / "validation_translated.jsonl"
TRAIN_AUGMENTED = OUTPUT_DIR / "train_augmented.jsonl"
VAL_AUGMENTED = OUTPUT_DIR / "validation_augmented.jsonl"

# Модель перевода NLLB-200
# Варианты:
#   facebook/nllb-200-distilled-600M  - легкая (~1.2GB VRAM, ~2.4GB на диске)
#   facebook/nllb-200-distilled-1.3B  - средняя (~3GB VRAM, ~5.5GB на диске) 
#   facebook/nllb-200-3.3B            - большая (~7GB VRAM, лучшее качество)
TRANSLATION_MODEL = "facebook/nllb-200-3.3B"  # легкая версия

# Путь для кеша моделей (в текущей директории)
CACHE_DIR = Path(__file__).parent / "models"

# Языковые коды NLLB
LANG_CODES = {
    "ru": "rus_Cyrl",
    "kk": "kaz_Cyrl"
}

# Батчинг для ускорения
BATCH_SIZE = 8
MAX_LENGTH = 512

# Количество воркеров для параллельной обработки
NUM_WORKERS = 1  # увеличь если много RAM

