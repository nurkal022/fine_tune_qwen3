#!/bin/bash
# Скрипт настройки окружения для fine-tuning Qwen3-32B
# RTX 5090 (Blackwell sm_120) + CUDA 13.0

set -e

echo "=============================================="
echo "Настройка окружения для RTX 5090 + Qwen3"
echo "=============================================="

# Проверка CUDA
echo ""
echo "[1/6] Проверка GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv

# Создание venv
echo ""
echo "[2/6] Создание виртуального окружения..."
python3 -m venv venv
source venv/bin/activate

# Обновление pip
echo ""
echo "[3/6] Обновление pip..."
pip install --upgrade pip setuptools wheel

# Установка PyTorch Nightly (поддержка Blackwell sm_120)
echo ""
echo "[4/6] Установка PyTorch Nightly с поддержкой RTX 5090..."
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126

# Проверка PyTorch + CUDA
echo ""
echo "Проверка PyTorch..."
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'GPU {i}: {props.name} ({props.total_memory // 1024**3} GB)')
"

# Установка зависимостей (БЕЗ Unsloth - он не работает с nightly)
echo ""
echo "[5/6] Установка transformers, peft, bitsandbytes..."
pip install transformers>=4.46.0 datasets>=3.0.0 trl>=0.12.0 peft>=0.13.0 accelerate>=1.0.0
pip install bitsandbytes>=0.44.0
pip install sentencepiece protobuf scipy einops tiktoken

# Финальная проверка
echo ""
echo "[6/6] Финальная проверка..."
python3 -c "
import torch
import transformers
import peft
import bitsandbytes

print('='*50)
print('Установленные версии:')
print('='*50)
print(f'PyTorch: {torch.__version__}')
print(f'Transformers: {transformers.__version__}')
print(f'PEFT: {peft.__version__}')
print(f'CUDA доступна: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'  GPU {i}: {props.name} ({props.total_memory // 1024**3} GB)')
"

echo ""
echo "=============================================="
echo "Готово!"
echo ""
echo "Для активации окружения:"
echo "  source venv/bin/activate"
echo ""
echo "Для запуска обучения:"
echo "  CUDA_VISIBLE_DEVICES=0 python train_32b_native.py"
echo "=============================================="
