"""
Unified fine-tuning script for Qwen3 models (4B/8B/14B)
All models use 4-bit quantization via Unsloth for uniform comparison.

Usage:
    python train.py --model 4b
    python train.py --model 8b --epochs 3
    python train.py --model 14b --lr 1e-4 --epochs 5
"""
import sys
import json
import argparse
import torch
import logging
from datetime import datetime
from pathlib import Path
from functools import partial

from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from transformers import TrainerCallback

from config import (
    TRAIN_FILE, VAL_FILE, TARGET_MODULES,
    LORA_R, LORA_ALPHA, LORA_DROPOUT,
)
from utils import log_gpu_info, format_dataset

# ============== MODEL CONFIGS ==============

MODEL_CONFIGS = {
    "4b": {
        "name": "unsloth/Qwen3-4B-unsloth-bnb-4bit",
        "batch_size": 4,
        "grad_accum": 4,   # effective batch = 16
    },
    "8b": {
        "name": "unsloth/Qwen3-8B-unsloth-bnb-4bit",
        "batch_size": 2,
        "grad_accum": 4,   # effective batch = 8
    },
    "14b": {
        "name": "unsloth/Qwen3-14B-unsloth-bnb-4bit",
        "batch_size": 1,
        "grad_accum": 8,   # effective batch = 8
    },
}


# ============== LOGGING CALLBACK ==============

class DetailedLoggingCallback(TrainerCallback):
    def __init__(self):
        self.start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = datetime.now()
        logger.info("=" * 60)
        logger.info("TRAINING STARTED")
        logger.info(f"Total steps: {state.max_steps}, Epochs: {args.num_train_epochs}")
        logger.info("=" * 60)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % args.logging_steps == 0:
            progress = state.global_step / state.max_steps * 100
            gpu_mem = gpu_total = gpu_percent = 0
            if torch.cuda.is_available():
                gpu_mem = torch.cuda.max_memory_reserved() / 1024**3
                gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                gpu_percent = gpu_mem / gpu_total * 100

            elapsed = (datetime.now() - self.start_time).total_seconds()
            if state.global_step > 0:
                remaining = (state.max_steps - state.global_step) * (elapsed / state.global_step)
                eta_str = f"{int(remaining // 3600)}h {int((remaining % 3600) // 60)}min"
            else:
                eta_str = "..."

            loss = state.log_history[-1].get('loss', 'N/A') if state.log_history else 'N/A'
            logger.info(
                f"Step {state.global_step}/{state.max_steps} ({progress:.1f}%) | "
                f"Loss: {loss} | GPU: {gpu_mem:.1f}/{gpu_total:.1f}GB ({gpu_percent:.0f}%) | "
                f"ETA: {eta_str}"
            )

    def on_save(self, args, state, control, **kwargs):
        logger.info(f"Checkpoint saved: {args.output_dir}/checkpoint-{state.global_step}")

    def on_epoch_end(self, args, state, control, **kwargs):
        logger.info(f"Epoch {int(state.epoch)} complete")

    def on_train_end(self, args, state, control, **kwargs):
        total = (datetime.now() - self.start_time).total_seconds()
        logger.info("=" * 60)
        logger.info(f"TRAINING COMPLETE in {int(total // 3600)}h {int((total % 3600) // 60)}min")
        if state.log_history:
            final_loss = [h.get('loss') for h in state.log_history if 'loss' in h]
            if final_loss:
                logger.info(f"Final loss: {final_loss[-1]:.4f}")
        logger.info("=" * 60)


# ============== MAIN ==============

def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune Qwen3 models with Unsloth')
    parser.add_argument('--model', type=str, required=True, choices=['4b', '8b', '14b'],
                        help='Model size: 4b, 8b, or 14b')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs (default: 3)')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate (default: 2e-4)')
    parser.add_argument('--batch-size', type=int, default=None, help='Override batch size')
    parser.add_argument('--grad-accum', type=int, default=None, help='Override gradient accumulation')
    parser.add_argument('--output-dir', type=str, default=None, help='Override output directory')
    parser.add_argument('--save-steps', type=int, default=1000, help='Save checkpoint every N steps')
    parser.add_argument('--max-seq-length', type=int, default=2048, help='Max sequence length')
    return parser.parse_args()


def main():
    args = parse_args()
    model_cfg = MODEL_CONFIGS[args.model]

    # Resolve config with CLI overrides
    model_name = model_cfg["name"]
    batch_size = args.batch_size or model_cfg["batch_size"]
    grad_accum = args.grad_accum or model_cfg["grad_accum"]
    output_dir = args.output_dir or f"outputs_{args.model}"
    lora_output = f"lora_qwen3_{args.model}"

    # Setup logging
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"train_{args.model}_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    global logger
    logger = logging.getLogger(__name__)

    # Log config
    config = {
        "model_size": args.model,
        "model_name": model_name,
        "max_seq_length": args.max_seq_length,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "effective_batch": batch_size * grad_accum,
        "num_epochs": args.epochs,
        "learning_rate": args.lr,
        "save_steps": args.save_steps,
        "output_dir": output_dir,
        "lora_output": lora_output,
    }

    logger.info("=" * 60)
    logger.info(f"Fine-tuning Qwen3-{args.model.upper()} (4-bit)")
    logger.info(f"Log: {log_file}")
    logger.info("=" * 60)
    logger.info("CONFIG:")
    for k, v in config.items():
        logger.info(f"  {k}: {v}")
    log_gpu_info()

    # Save config JSON
    config_file = log_dir / f"config_{args.model}_{timestamp}.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # 1. Load model
    logger.info("\n[1/5] Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    log_gpu_info()

    # 2. LoRA
    logger.info("\n[2/5] Applying LoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Trainable: {trainable:,} ({trainable/total_params*100:.2f}%)")

    # 3. Dataset
    logger.info("\n[3/5] Loading dataset...")
    dataset = load_dataset("json", data_files={"train": TRAIN_FILE, "validation": VAL_FILE})
    train_dataset = dataset["train"].map(
        partial(format_dataset, eos_token=tokenizer.eos_token), batched=True
    )
    logger.info(f"  Train: {len(train_dataset):,}, Val: {len(dataset['validation']):,}")

    effective_batch = batch_size * grad_accum
    steps_per_epoch = len(train_dataset) // effective_batch
    logger.info(f"  Steps/epoch: {steps_per_epoch:,}, Total: ~{steps_per_epoch * args.epochs:,}")

    # 4. Trainer
    logger.info("\n[4/5] Initializing trainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        callbacks=[DetailedLoggingCallback()],
        args=SFTConfig(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            warmup_ratio=0.03,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            logging_steps=50,
            eval_strategy="no",
            save_strategy="steps",
            save_steps=args.save_steps,
            optim="adamw_8bit",
            weight_decay=0.001,
            lr_scheduler_type="cosine",
            seed=3407,
            report_to="none",
            bf16=True,
        ),
    )
    log_gpu_info()

    # 5. Train
    logger.info("\n[5/5] Starting training...")
    stats = trainer.train()

    runtime = stats.metrics['train_runtime']
    logger.info(f"\nTime: {int(runtime // 3600)}h {int((runtime % 3600) // 60)}min")
    logger.info(f"Train loss: {stats.metrics.get('train_loss', 'N/A')}")
    log_gpu_info()

    # Save LoRA
    logger.info(f"\nSaving LoRA to {lora_output}/...")
    model.save_pretrained(lora_output)
    tokenizer.save_pretrained(lora_output)

    # Save training metrics
    metrics_file = log_dir / f"metrics_{args.model}_{timestamp}.json"
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(stats.metrics, f, indent=2, ensure_ascii=False)

    logger.info("DONE!")


if __name__ == "__main__":
    main()
