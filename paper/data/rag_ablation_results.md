# RAG Ablation Results (200 samples)

## FT Model (Qwen3-4B FT)

| Configuration | BERTScore | Citation Acc. | Halluc. Rate | Key Info |
|---|---|---|---|---|
| FT only | 89.8% | 81.9% | 26.5% | 75.8% |
| FT + RAG (top-1) | 88.3% | 77.2% | 27.4% | 69.5% |
| FT + RAG (top-3) | 88.0% | 77.6% | 29.9% | 72.5% |
| FT + RAG (top-5) | 88.1% | 78.4% | 25.3% | 70.7% |

## Base Model (Qwen3-4B Base)

| Configuration | BERTScore | Citation Acc. | Halluc. Rate | Key Info |
|---|---|---|---|---|
| Base only | 82.6% | 73.5% | 21.8% | 69.5% |
| Base + RAG (top-1) | 82.8% | 75.6% | 35.9% | 72.0% |
| Base + RAG (top-3) | 83.1% | 78.1% | 30.9% | 72.6% |
| Base + RAG (top-5) | 83.2% | 80.3% | 35.9% | 73.7% |

## Key Finding
RAG hurts FT models (-1.7% BERTScore, -3.5% CitAcc) but helps Base models (+6.8% CitAcc).
Fine-tuning internalizes domain knowledge, making RAG redundant.
