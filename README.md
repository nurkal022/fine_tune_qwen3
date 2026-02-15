# Fine-tuning Qwen3 for Kazakhstan Legal Domain

Fine-tuning Qwen3 models (8B/14B/32B) on ~57,000 examples of Kazakhstan legislation using LoRA.

## Results (Qwen3-8B)

| Metric | Value |
|--------|-------|
| Initial Loss | 1.8544 |
| Final Loss | 0.5734 |
| Loss Reduction | 69.1% |
| Key Info Score | 77.45% |
| Training Time | ~24 hours |
| GPU | RTX 5080 (16GB) |

See [RESULTS.md](RESULTS.md) for detailed benchmarks and plots.

## Project Structure

```
config.py                 # Shared config: ALPACA_PROMPT, paths, LoRA params
utils.py                  # Shared utilities: gpu_info, dataset formatting
prepare_data.py           # Dataset preparation & merging

train_8b.py               # Qwen3-8B  (4-bit Unsloth, RTX 5080 16GB)
train_14b.py              # Qwen3-14B (bf16 Unsloth, RTX 5090 32GB)
train_32b_native.py       # Qwen3-32B (4-bit QLoRA, native transformers)
train_32b_deepspeed.py    # Qwen3-32B (bf16 DeepSpeed ZeRO-3, 2x RTX 5090)

chat.py                   # Interactive chat with trained model
inference.py              # Batch inference & test examples
benchmark.py              # Model evaluation (ROUGE, BLEU, Token F1)
generate_plots.py         # Training loss & benchmark visualization
export_model.py           # Export to GGUF / 16bit / 4bit

setup_env.sh              # Environment setup for RTX 5090
run_training.sh           # Training launcher

combined_data/            # Training dataset (57K train / 6K val)
finetune_dataset/         # Original dataset source
```

## Requirements

- Python 3.10+
- CUDA 12.0+
- GPU: RTX 5080 (16GB) for 8B, RTX 5090 (32GB) for 14B/32B

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install unsloth
pip install -r requirements.txt
```

## Training

```bash
# 8B model (4-bit, ~16GB VRAM)
python train_8b.py

# 14B model (bf16, ~30GB VRAM)
python train_14b.py

# 32B model (4-bit QLoRA, ~30GB VRAM)
python train_32b_native.py

# 32B model (DeepSpeed, 2x GPU)
deepspeed --num_gpus=2 train_32b_deepspeed.py
```

## Inference

```bash
# Interactive chat
python chat.py --model lora_qwen3_8b

# Benchmark
python benchmark.py

# Plots
python generate_plots.py
```

## Data Format

JSONL with Alpaca format:

```json
{
  "instruction": "Ответь на вопрос по казахстанскому законодательству.",
  "input": "Можно ли заключить договор ГПХ с указанием цены работы за сутки?",
  "output": "Согласно законодательству РК..."
}
```

## License

MIT
