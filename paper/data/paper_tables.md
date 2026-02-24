# Experiment Results

## Table 1: Automated Metrics (500 samples, max_new_tokens=1024)

| Model | BERTScore F1 | ROUGE-L | Citation Acc. | Halluc. Rate | Key Info | Latency (s) |
|---|---|---|---|---|---|---|
| Qwen3-14B FT | 90.1% ± 0.3 | 8.5% ± 1.9 | 80.5% ± 3.0 | 27.5% ± 3.6 | 73.7% ± 2.5 | 3.82 |
| Qwen3-8B FT | 89.9% ± 0.3 | 7.9% ± 1.8 | 78.8% ± 3.3 | 29.3% ± 3.8 | 72.8% ± 2.6 | 4.59 |
| Qwen3-4B FT | 89.6% ± 0.3 | 7.3% ± 1.6 | 79.7% ± 3.2 | 28.1% ± 3.8 | 72.6% ± 2.6 | 3.34 |
| GPT-4o | 87.2% ± 0.2 | 5.8% ± 1.5 | 76.3% ± 3.6 | 83.0% ± 3.3 | 75.6% ± 2.4 | 2.94 |
| GPT-4o-mini | 86.7% ± 0.2 | 6.1% ± 1.4 | 75.1% ± 3.7 | 89.9% ± 2.6 | 76.1% ± 2.4 | 3.86 |
| Qwen3-14B Base | 81.8% ± 0.2 | 0.2% ± 0.1 | 73.1% ± 3.8 | 32.5% ± 4.2 | 76.9% ± 2.6 | 19.35 |
| Qwen3-8B Base | 81.7% ± 0.2 | 2.5% ± 0.7 | 73.3% ± 3.8 | 23.6% ± 3.8 | 73.1% ± 2.8 | 30.39 |
| Qwen3-4B Base | 81.0% ± 0.3 | 1.6% ± 0.7 | 73.5% ± 3.8 | 24.1% ± 3.6 | 69.8% ± 2.8 | 21.01 |

Notes:
- 95% CI via bootstrap (1000 iterations), random.seed(42)
- 14B on RTX 5090 (32GB), 4B/8B on RTX 5080 Laptop (16GB)
- GPT-4o/mini via OpenAI API (max_tokens=1024)

## Table 2: Human Evaluation by Legal Experts (50 questions × 6 models)

| Model | Correctness (1-5) | Completeness (1-5) | Relevance (1-5) | Hallucinations |
|---|---|---|---|---|
| Qwen3-14B FT | **3.90** | **4.70** | **3.95** | 0/50 (0%) |
| Qwen3-8B FT | 3.70 | 4.66 | 3.73 | 1/50 (2%) |
| Qwen3-4B FT | 3.00 | 3.58 | 3.50 | 0/50 (0%) |
| Qwen3-14B Base | 3.42 | 1.90 | 3.50 | 0/50 (0%) |
| Qwen3-4B Base | 3.42 | 1.96 | 3.48 | 2/50 (4%) |
| Qwen3-8B Base | 3.24 | 1.86 | 3.26 | 1/50 (2%) |

| Aggregate | Correctness | Completeness | Relevance | Hallucinations |
|---|---|---|---|---|
| FT (avg) | 3.53 | 4.31 | 3.73 | 1/150 (0.7%) |
| Base (avg) | 3.36 | 1.91 | 3.41 | 3/150 (2.0%) |

Notes:
- Blind evaluation: answers shuffled as A-F, evaluators unaware of model identity
- Evaluated by practicing lawyers specializing in Kazakhstan law
- Criteria: Correctness (legal accuracy), Completeness (exhaustiveness), Relevance (on-topic), Hallucination (fabricated legal norms)
