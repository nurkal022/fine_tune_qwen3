#!/bin/bash
# Скрипт запуска обучения Qwen3-32B

cd "$(dirname "$0")"

# Активация окружения
source venv/bin/activate

# Запуск обучения на GPU 0
# Можно изменить на GPU 1 или убрать для использования всех GPU
CUDA_VISIBLE_DEVICES=0 python train_32b_native.py
