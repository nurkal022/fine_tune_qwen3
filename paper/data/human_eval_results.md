# Human Evaluation Results (Updated Feb 24, 2026)

## Protocol
- 50 questions × 6 models = 300 blind evaluations
- Evaluators: Practicing lawyers (Kazakhstan law)
- Criteria: Correctness (1-5), Completeness (1-5), Relevance (1-5), Hallucination (yes/no)
- Answers shuffled as A-F per question (random.seed(42))

## Results by Model

| Model | Correctness (1-5) | Completeness (1-5) | Relevance (1-5) | Hallucinations |
|---|---|---|---|---|
| Qwen3-14B FT | **3.90** | **4.70** | **3.95** | 0/50 (0%) |
| Qwen3-8B FT | 3.70 | 4.66 | 3.73 | 1/50 (2%) |
| Qwen3-4B FT | 3.00 | 3.58 | 3.50 | 0/50 (0%) |
| Qwen3-14B Base | 3.42 | 1.90 | 3.50 | 0/50 (0%) |
| Qwen3-4B Base | 3.42 | 1.96 | 3.48 | 2/50 (4%) |
| Qwen3-8B Base | 3.24 | 1.86 | 3.26 | 1/50 (2%) |

## Aggregate: FT vs Base

| Aggregate | Correctness | Completeness | Relevance | Hallucinations |
|---|---|---|---|---|
| FT (avg) | 3.53 | 4.31 | 3.73 | 1/150 (0.7%) |
| Base (avg) | 3.36 | 1.91 | 3.41 | 3/150 (2.0%) |
