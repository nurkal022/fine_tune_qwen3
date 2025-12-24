#!/bin/bash
# Полный пайплайн перевода и объединения датасета

set -e

cd "$(dirname "$0")"

echo "=========================================="
echo "АУГМЕНТАЦИЯ ДАТАСЕТА ПЕРЕВОДОМ kk↔ru"
echo "=========================================="

# Активация conda (если нужно)
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate ai

echo ""
echo "[Step 1/2] Перевод датасета..."
python translate_dataset.py

echo ""
echo "[Step 2/2] Объединение датасетов..."
python merge_datasets.py

echo ""
echo "=========================================="
echo "ГОТОВО!"
echo "=========================================="
echo ""
echo "Результаты в папке: output/"
echo "  - train_translated.jsonl   (только переводы)"
echo "  - validation_translated.jsonl"
echo "  - train_augmented.jsonl    (оригинал + переводы)"
echo "  - validation_augmented.jsonl"

