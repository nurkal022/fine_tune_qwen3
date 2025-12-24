"""
Объединение оригинального и переведённого датасетов
"""
import json
import random
from pathlib import Path

from config import (
    TRAIN_FILE, VAL_FILE, OUTPUT_DIR,
    TRAIN_TRANSLATED, VAL_TRANSLATED,
    TRAIN_AUGMENTED, VAL_AUGMENTED
)


def load_jsonl(path: Path) -> list:
    """Загрузка JSONL файла"""
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def save_jsonl(records: list, path: Path):
    """Сохранение в JSONL"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')


def merge_files(original_path: Path, translated_path: Path, output_path: Path):
    """Объединение оригинала и перевода"""
    print(f"\nОбъединение:")
    print(f"  Оригинал: {original_path}")
    print(f"  Перевод: {translated_path}")
    
    original = load_jsonl(original_path)
    translated = load_jsonl(translated_path)
    
    print(f"  Оригинал: {len(original)} записей")
    print(f"  Перевод: {len(translated)} записей")
    
    # Объединяем
    combined = original + translated
    
    # Перемешиваем
    random.seed(42)
    random.shuffle(combined)
    
    # Сохраняем
    save_jsonl(combined, output_path)
    
    print(f"  Итого: {len(combined)} записей → {output_path}")


def main():
    print("="*50)
    print("ОБЪЕДИНЕНИЕ ДАТАСЕТОВ")
    print("="*50)
    
    # Train
    if TRAIN_FILE.exists() and TRAIN_TRANSLATED.exists():
        merge_files(TRAIN_FILE, TRAIN_TRANSLATED, TRAIN_AUGMENTED)
    else:
        print(f"Файлы train не найдены!")
    
    # Validation
    if VAL_FILE.exists() and VAL_TRANSLATED.exists():
        merge_files(VAL_FILE, VAL_TRANSLATED, VAL_AUGMENTED)
    else:
        print(f"Файлы validation не найдены!")
    
    print("\n" + "="*50)
    print("ГОТОВО!")
    print("="*50)
    
    # Статистика
    if TRAIN_AUGMENTED.exists():
        train_count = sum(1 for _ in open(TRAIN_AUGMENTED))
        print(f"\nАугментированный Train: {train_count} записей")
    
    if VAL_AUGMENTED.exists():
        val_count = sum(1 for _ in open(VAL_AUGMENTED))
        print(f"Аугментированный Val: {val_count} записей")


if __name__ == "__main__":
    main()

