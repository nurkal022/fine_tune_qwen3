# Fine-tuning Qwen3 for Kazakhstan Legal Domain

Fine-tuning Qwen3 models (4B/8B/14B/32B) on ~57,000 examples of Kazakhstan legislation using LoRA + Unsloth.

## Experiments

| # | Experiment | Script | What it produces |
|---|-----------|--------|-----------------|
| 1 | Unified Benchmark | `benchmark.py` | BERTScore, ROUGE-L, Citation Acc, Hallucination Rate |
| 2 | Scaling Analysis | `compare_results.py` | Base vs FT comparison table across 4B/8B/14B |
| 3 | RAG Ablation | `rag_benchmark.py` | FT only vs FT+RAG (top-1/3/5) |
| 4 | Domain/Language Breakdown | `benchmark.py` + `compare_results.py` | Per-domain and RU vs KZ analysis |
| 5 | Human Evaluation | `prepare_human_eval.py` | Blind evaluation sheets for lawyers |

## Project Structure

```
config.py                 # Shared config: ALPACA_PROMPT, paths, LoRA params
utils.py                  # Shared utilities: gpu_info, dataset formatting

train.py                  # Unified training (4B/8B/14B via Unsloth 4-bit)
train_32b_native.py       # Qwen3-32B (4-bit QLoRA, native transformers)
train_32b_deepspeed.py    # Qwen3-32B (bf16 DeepSpeed ZeRO-3, 2x GPU)

benchmark.py              # Exp 1+4: Evaluation with domain/language breakdown
compare_results.py        # Exp 2+4: Scaling tables, cross-model comparison
rag_benchmark.py          # Exp 3: RAG ablation (FT vs FT+RAG)
prepare_human_eval.py     # Exp 5: Blind evaluation sheets for lawyers
analyze_dataset.py        # Dataset statistics for paper

chat.py                   # Interactive chat with trained model
inference.py              # Batch inference & test examples
generate_plots.py         # Training loss & benchmark visualization
export_model.py           # Export to GGUF / 16bit / 4bit
prepare_data.py           # Dataset preparation & merging

setup_env.sh              # Environment setup for RTX 5090
run_training.sh           # Training launcher
combined_data/            # Training dataset (57K train / 6K val)
results/                  # Benchmark result JSONs
human_eval/               # Evaluation sheets for lawyers
```

## Quick Start

```bash
# Setup
conda activate ai
pip install -r requirements.txt

# Dataset stats (Exp 4 data section)
python analyze_dataset.py

# Train all sizes
python train.py --model 4b --epochs 3
python train.py --model 8b --epochs 3
python train.py --model 14b --epochs 3

# Baseline benchmarks
python benchmark.py --baseline unsloth/Qwen3-4B-unsloth-bnb-4bit --samples 100
python benchmark.py --baseline unsloth/Qwen3-8B-unsloth-bnb-4bit --samples 100
python benchmark.py --baseline unsloth/Qwen3-14B-unsloth-bnb-4bit --samples 100

# Fine-tuned benchmarks
python benchmark.py --model lora_qwen3_4b --samples 100
python benchmark.py --model lora_qwen3_8b --samples 100
python benchmark.py --model lora_qwen3_14b --samples 100

# Compare all results (Exp 2 + 4)
python compare_results.py results/ --markdown results/paper_tables.md

# RAG ablation (Exp 3) — requires legal corpus
python rag_benchmark.py --model lora_qwen3_8b --corpus legal_docs/ --top-k 1 3 5

# Human eval sheets (Exp 5)
python prepare_human_eval.py --from-results results/*.json --questions 50
```

## License

MIT
