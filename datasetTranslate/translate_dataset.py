"""
Основной скрипт перевода датасета kk↔ru
"""
import json
import os
from pathlib import Path
from tqdm import tqdm

from config import (
    TRAIN_FILE, VAL_FILE, OUTPUT_DIR,
    TRAIN_TRANSLATED, VAL_TRANSLATED
)
from translator import Translator


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


def translate_file(translator: Translator, input_path: Path, output_path: Path):
    """Перевод одного файла"""
    print(f"\n{'='*50}")
    print(f"Перевод: {input_path.name}")
    print(f"{'='*50}")
    
    records = load_jsonl(input_path)
    print(f"Загружено {len(records)} записей")
    
    translated = []
    errors = 0
    
    # Статистика по языкам
    lang_stats = {"kk": 0, "ru": 0}
    
    for record in tqdm(records, desc="Перевод"):
        try:
            trans = translator.translate_record(record)
            lang_stats[trans["_original_lang"]] += 1
            
            # Убираем служебные поля
            clean_record = {
                "instruction": trans["instruction"],
                "input": trans["input"],
                "output": trans["output"],
            }
            translated.append(clean_record)
            
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"\nОшибка: {e}")
            continue
    
    save_jsonl(translated, output_path)
    
    print(f"\nСтатистика:")
    print(f"  Казахских записей: {lang_stats['kk']} → переведено на русский")
    print(f"  Русских записей: {lang_stats['ru']} → переведено на казахский")
    print(f"  Ошибок: {errors}")
    print(f"  Сохранено: {len(translated)} записей в {output_path}")


def main():
    print("="*50)
    print("ПЕРЕВОД ДАТАСЕТА kk↔ru")
    print("="*50)
    
    # Создаём директорию вывода
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Инициализация переводчика
    translator = Translator()
    
    # Перевод train
    if TRAIN_FILE.exists():
        translate_file(translator, TRAIN_FILE, TRAIN_TRANSLATED)
    else:
        print(f"Файл не найден: {TRAIN_FILE}")
    
    # Перевод validation
    if VAL_FILE.exists():
        translate_file(translator, VAL_FILE, VAL_TRANSLATED)
    else:
        print(f"Файл не найден: {VAL_FILE}")
    
    print("\n" + "="*50)
    print("ПЕРЕВОД ЗАВЕРШЁН!")
    print("="*50)
    print(f"\nФайлы сохранены в: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

