#!/bin/bash
# =============================================================
# Полный эксперимент для 14B: обучение + бенчмарки
# Запуск: bash run_14b_experiment.sh
#
# Перед запуском:
#   1. git clone <repo-url> && cd fineTuneQwen
#   2. conda activate ai  (или ваше окружение)
#   3. pip install -r requirements.txt
#   4. Убедитесь что combined_data/train.jsonl и validation.jsonl на месте
# =============================================================
set -e

echo "============================================================"
echo "  14B EXPERIMENT PIPELINE"
echo "  $(date)"
echo "============================================================"

# Проверяем что данные на месте
if [ ! -f "combined_data/train.jsonl" ]; then
    echo "ERROR: combined_data/train.jsonl not found!"
    echo "Убедитесь что данные скопированы."
    exit 1
fi
if [ ! -f "combined_data/validation.jsonl" ]; then
    echo "ERROR: combined_data/validation.jsonl not found!"
    exit 1
fi

# Проверяем GPU
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_mem/1e9:.1f}GB')" || {
    echo "ERROR: No GPU detected!"; exit 1
}

echo ""
echo "============================================================"
echo "  STEP 1/4: Training 14B model (3 epochs)"
echo "============================================================"
PYTHONUNBUFFERED=1 python train.py --model 14b --epochs 3 2>&1 | tee logs/train_14b_run.log

echo ""
echo "============================================================"
echo "  STEP 2/4: Baseline 14B benchmark (100 samples)"
echo "============================================================"
mkdir -p results
PYTHONUNBUFFERED=1 python benchmark.py \
    --baseline "unsloth/Qwen3-14B-unsloth-bnb-4bit" \
    --samples 100 \
    --output results/baseline_14b.json \
    2>&1 | tee logs/benchmark_baseline_14b.log

echo ""
echo "============================================================"
echo "  STEP 3/4: Fine-tuned 14B benchmark (100 samples)"
echo "============================================================"
PYTHONUNBUFFERED=1 python benchmark.py \
    --model lora_qwen3_14b \
    --samples 100 \
    --output results/benchmark_14b_ft.json \
    2>&1 | tee logs/benchmark_ft_14b.log

echo ""
echo "============================================================"
echo "  STEP 4/4: Summary"
echo "============================================================"
echo ""
echo "Results files:"
ls -la results/baseline_14b.json results/benchmark_14b_ft.json 2>/dev/null
echo ""

# Показать ключевые метрики
python -c "
import json

for label, path in [('Baseline 14B', 'results/baseline_14b.json'), ('FT 14B', 'results/benchmark_14b_ft.json')]:
    with open(path) as f:
        d = json.load(f)
    m = d['avg_metrics']
    print(f'{label}:')
    print(f'  BERTScore F1:   {m.get(\"bertscore_f1\", 0)*100:.2f}%')
    print(f'  ROUGE-L:        {m.get(\"rougeL\", 0)*100:.2f}%')
    print(f'  Citation Acc:   {m.get(\"citation_accuracy\", 0)*100:.2f}%')
    print(f'  Halluc Rate:    {m.get(\"hallucination_rate\", 0)*100:.2f}%')
    print(f'  Key Info:       {m.get(\"key_info\", 0)*100:.2f}%')
    print(f'  Latency:        {m.get(\"gen_time\", 0):.2f}s')
    print()
"

echo "============================================================"
echo "  DONE! $(date)"
echo "============================================================"
echo ""
echo "Скопируйте результаты обратно:"
echo "  scp results/baseline_14b.json results/benchmark_14b_ft.json user@main-pc:/path/to/fineTuneQwen/results/"
echo ""
echo "Или закоммитьте:"
echo "  git add results/ logs/ lora_qwen3_14b/"
echo "  git commit -m 'Add 14B training results and benchmarks'"
echo "  git push"
