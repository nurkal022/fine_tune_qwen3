#!/bin/bash

# Активация conda окружения и запуск обучения
# Использование: ./run_training.sh

set -e

echo "=========================================="
echo "Fine-tuning Qwen3-14B"
echo "=========================================="

# Переход в директорию проекта
cd "$(dirname "$0")"

# Активация conda
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || \
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || \
source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || \
echo "Conda activation failed, assuming already activated"

conda activate ai

echo ""
echo "[Step 1/2] Подготовка данных..."
python prepare_data.py

echo ""
echo "[Step 2/2] Запуск обучения..."
python train.py

echo ""
echo "=========================================="
echo "Обучение завершено!"
echo "=========================================="

