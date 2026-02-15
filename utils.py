"""
Общие утилиты для fine-tuning проекта
"""
import torch
import logging

from config import ALPACA_PROMPT

logger = logging.getLogger(__name__)


def print_gpu_info():
    """Вывод информации о GPU (все устройства)"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu = torch.cuda.get_device_properties(i)
            allocated = round(torch.cuda.memory_allocated(i) / 1024**3, 2)
            reserved = round(torch.cuda.memory_reserved(i) / 1024**3, 2)
            total = round(gpu.total_memory / 1024**3, 2)
            print(f"GPU {i}: {gpu.name} | {allocated}/{total} GB allocated, {reserved} GB reserved")
    else:
        print("CUDA not available!")


def log_gpu_info():
    """Логирование информации о GPU через logging"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu = torch.cuda.get_device_properties(i)
            allocated = round(torch.cuda.memory_allocated(i) / 1024**3, 2)
            reserved = round(torch.cuda.memory_reserved(i) / 1024**3, 2)
            total = round(gpu.total_memory / 1024**3, 2)
            logger.info(f"GPU {i}: {gpu.name} | {allocated}/{total} GB allocated, {reserved} GB reserved")
    else:
        logger.warning("CUDA not available!")


def format_dataset(examples, eos_token):
    """Форматирование датасета в Alpaca формат с EOS токеном"""
    texts = []
    for instr, inp, out in zip(examples["instruction"], examples["input"], examples["output"]):
        texts.append(ALPACA_PROMPT.format(instr, inp, out) + eos_token)
    return {"text": texts}
