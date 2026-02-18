#!/bin/bash
# =============================================================
# 14B бенчмарки: 500 samples + bootstrap CI
# Запуск: bash run_14b_benchmark.sh
# (модель lora_qwen3_14b уже обучена)
# =============================================================
set -e
mkdir -p logs results

echo "============================================================"
echo "  14B BENCHMARK (500 samples + CI)"
echo "  $(date)"
echo "============================================================"

# Проверяем что модель на месте
if [ ! -d "lora_qwen3_14b" ]; then
    echo "ERROR: lora_qwen3_14b/ not found!"
    exit 1
fi

echo ""
echo "  STEP 1/2: Baseline 14B (500 samples)"
echo "============================================================"
PYTHONUNBUFFERED=1 python benchmark.py \
    --baseline "unsloth/Qwen3-14B-unsloth-bnb-4bit" \
    --samples 500 \
    --output results/baseline_14b.json \
    2>&1 | tee logs/benchmark_baseline_14b_500.log

echo ""
echo "============================================================"
echo "  STEP 2/2: Fine-tuned 14B (500 samples)"
echo "============================================================"
PYTHONUNBUFFERED=1 python benchmark.py \
    --model lora_qwen3_14b \
    --samples 500 \
    --output results/benchmark_14b_ft.json \
    2>&1 | tee logs/benchmark_ft_14b_500.log

echo ""
echo "============================================================"
echo "  DONE! $(date)"
echo "============================================================"
echo ""
echo "Скопируйте результаты:"
echo "  scp results/baseline_14b.json results/benchmark_14b_ft.json user@main-pc:/path/to/fineTuneQwen/results/"
