"""
Скрипт объединения датасетов для fine-tuning
"""
import json
import os
import random
from pathlib import Path
from config import DataConfig

def load_jsonl(filepath: str) -> list:
    """Загрузка JSONL файла"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def normalize_record(record: dict) -> dict:
    """Нормализация записи - убираем metadata, оставляем только нужные поля"""
    return {
        "instruction": record.get("instruction", ""),
        "input": record.get("input", ""),
        "output": record.get("output", "")
    }

def save_jsonl(data: list, filepath: str):
    """Сохранение в JSONL формат"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        for record in data:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

def main():
    config = DataConfig()
    random.seed(42)
    
    print("=" * 50)
    print("Подготовка объединенного датасета")
    print("=" * 50)
    
    # 1. Загрузка finetune_dataset
    print("\n[1/4] Загрузка finetune_dataset...")
    train_data = load_jsonl(config.finetune_train)
    val_data = load_jsonl(config.finetune_val)
    print(f"  Train: {len(train_data)} записей")
    print(f"  Val: {len(val_data)} записей")
    
    # 2. Загрузка data_final_pars
    print("\n[2/4] Загрузка data_final_pars...")
    final_pars_data = load_jsonl(config.data_final_pars)
    print(f"  Всего: {len(final_pars_data)} записей")
    
    # Разделение на train/val
    random.shuffle(final_pars_data)
    split_idx = int(len(final_pars_data) * (1 - config.val_split_ratio))
    final_pars_train = final_pars_data[:split_idx]
    final_pars_val = final_pars_data[split_idx:]
    print(f"  Train split: {len(final_pars_train)} записей")
    print(f"  Val split: {len(final_pars_val)} записей")
    
    # 3. Нормализация и объединение
    print("\n[3/4] Нормализация и объединение...")
    combined_train = [normalize_record(r) for r in train_data] + \
                     [normalize_record(r) for r in final_pars_train]
    combined_val = [normalize_record(r) for r in val_data] + \
                   [normalize_record(r) for r in final_pars_val]
    
    # Перемешиваем
    random.shuffle(combined_train)
    random.shuffle(combined_val)
    
    print(f"  Объединенный Train: {len(combined_train)} записей")
    print(f"  Объединенный Val: {len(combined_val)} записей")
    
    # 4. Сохранение
    print("\n[4/4] Сохранение...")
    save_jsonl(combined_train, config.combined_train)
    save_jsonl(combined_val, config.combined_val)
    print(f"  Сохранено: {config.combined_train}")
    print(f"  Сохранено: {config.combined_val}")
    
    print("\n" + "=" * 50)
    print("Готово!")
    print("=" * 50)
    
    # Статистика
    print("\nСтатистика объединенного датасета:")
    total = len(combined_train) + len(combined_val)
    print(f"  Всего записей: {total}")
    print(f"  Train: {len(combined_train)} ({len(combined_train)/total*100:.1f}%)")
    print(f"  Val: {len(combined_val)} ({len(combined_val)/total*100:.1f}%)")

if __name__ == "__main__":
    main()

