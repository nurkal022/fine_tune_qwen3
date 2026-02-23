# Experiment Results

## Table 1: Model Comparison (max_new_tokens=1024, no truncation)

| Model | BERTScore F1 | ROUGE-L | Citation Acc. | Halluc. Rate | Key Info | Latency (s) |
|---|---|---|---|---|---|---|
| Qwen3-14B FT | 90.1% ± 0.3 | 9.1% ± 1.9 | 80.5% ± 3.1 | 27.6% ± 3.7 | 73.4% ± 2.5 | 3.20 |
| Qwen3-4B FT | 89.6% ± 0.3 | 7.3% ± 1.6 | 79.7% ± 3.2 | 28.1% ± 3.8 | 72.6% ± 2.6 | 3.34 |
| Qwen3-8B FT | 89.9% ± 0.3 | 7.9% ± 1.8 | 78.8% ± 3.3 | 29.3% ± 3.8 | 72.8% ± 2.6 | 4.59 |
| Qwen3-14B Base | 81.8% ± 0.2 | 0.1% ± 0.0 | 73.1% ± 3.7 | 15.4% ± 3.1 | 63.5% ± 3.2 | 10.36 |
| Qwen3-4B Base | 81.0% ± 0.3 | 1.6% ± 0.7 | 73.5% ± 3.8 | 24.1% ± 3.6 | 69.8% ± 2.8 | 21.01 |
| Qwen3-8B Base | 81.7% ± 0.2 | 2.5% ± 0.7 | 73.3% ± 3.8 | 23.6% ± 3.8 | 73.1% ± 2.8 | 30.39 |

Notes:
- 500 samples per model, random.seed(42), 95% CI via bootstrap (1000 iterations)
- 14B benchmarks run on RTX 5090 (32GB), 4B/8B on RTX 5080 Laptop (16GB)
- FT = LoRA fine-tuned, Base = pre-trained baseline (no fine-tuning)
