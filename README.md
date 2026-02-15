# Fine-tuning Qwen3 for Kazakhstan Legal Domain

Fine-tuning Qwen3 models (4B/8B/14B/32B) on ~57,000 examples of Kazakhstan legislation using LoRA + Unsloth.

## Project Structure

```
config.py                 # Shared config: ALPACA_PROMPT, paths, LoRA params
utils.py                  # Shared utilities: gpu_info, dataset formatting

train.py                  # Unified training (4B/8B/14B via Unsloth 4-bit)
train_32b_native.py       # Qwen3-32B (4-bit QLoRA, native transformers)
train_32b_deepspeed.py    # Qwen3-32B (bf16 DeepSpeed ZeRO-3, 2x GPU)

benchmark.py              # Model evaluation (BERTScore, ROUGE, Citation Acc)
analyze_dataset.py        # Dataset statistics for paper
chat.py                   # Interactive chat with trained model
inference.py              # Batch inference & test examples
generate_plots.py         # Training loss & benchmark visualization
export_model.py           # Export to GGUF / 16bit / 4bit
prepare_data.py           # Dataset preparation & merging

setup_env.sh              # Environment setup for RTX 5090
run_training.sh           # Training launcher
combined_data/            # Training dataset (57K train / 6K val)
finetune_dataset/         # Original dataset source
```

## Requirements

- Python 3.10+
- CUDA 12.0+
- GPU: RTX 5090 (32GB) recommended

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install unsloth
pip install -r requirements.txt
```

## Training

```bash
# Qwen3-4B (4-bit, ~8GB VRAM)
python train.py --model 4b

# Qwen3-8B (4-bit, ~16GB VRAM)
python train.py --model 8b --epochs 3

# Qwen3-14B (4-bit, ~24GB VRAM)
python train.py --model 14b --epochs 3 --lr 1e-4

# Qwen3-32B (native QLoRA, ~30GB VRAM)
python train_32b_native.py

# Qwen3-32B (DeepSpeed ZeRO-3, 2x GPU)
deepspeed --num_gpus=2 train_32b_deepspeed.py
```

## Evaluation

```bash
# Benchmark fine-tuned model
python benchmark.py --model lora_qwen3_8b --samples 100

# Benchmark baseline (untuned model)
python benchmark.py --baseline unsloth/Qwen3-8B-unsloth-bnb-4bit --samples 100

# Dataset statistics
python analyze_dataset.py

# Interactive chat
python chat.py --model lora_qwen3_8b
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
