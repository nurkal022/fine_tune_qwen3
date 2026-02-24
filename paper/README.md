# Paper Materials

All materials for the paper "Fine-Tuning Qwen3 for Kazakhstan Legal Domain".

## Structure

```
paper/
├── paper_draft.md          # Full paper draft (Sections 1-8)
├── README.md               # This file
├── figures/
│   ├── fig1_scaling_analysis.png     # Model size comparison (4B/8B/14B)
│   ├── fig2_rag_ablation.png         # RAG effectiveness chart
│   ├── fig3_domain_breakdown.png     # Citation accuracy by legal domain
│   ├── fig4_language_comparison.png  # Performance: Russian vs Kazakh
│   └── fig5_training_loss.png        # Training loss curve
└── data/
    ├── paper_tables.md               # All formatted tables
    ├── human_eval_results.md         # Human evaluation detailed results
    ├── rag_ablation_results.md       # RAG experiment results
    ├── dataset_stats.json            # Dataset statistics
    └── config.py                     # Training configuration
```

## Paper Sections Status

| Section | Status |
|---------|--------|
| Abstract | Done |
| 1. Introduction | Done |
| 2. Related Work | **TODO** (author) |
| 3. Dataset | Done |
| 4. Methodology | Done |
| 5. Evaluation Methodology | Done |
| 6. Results (6.1-6.5) | Done |
| 7. Discussion | Done |
| 8. Conclusion | Done |
| References | **TODO** (author) |

## Key Numbers for Quick Reference

- Dataset: 63,114 samples (56,802 train / 6,312 val)
- Languages: Russian 76.2%, Kazakh 23.8%
- Legal domains: 11
- Best automated: Qwen3-14B FT (BERTScore 90.1%, CitAcc 80.5%)
- Best cost-effective: Qwen3-4B FT (BERTScore 89.6%, CitAcc 79.7%, 3.34s, 7GB VRAM)
- Human eval FT avg: Correctness 3.53, Completeness 4.31, Relevance 3.73
- GPT-4o: BERTScore 87.2%, Halluc 83.0% (3x worse than FT)
- RAG: Helps base (+6.8% CitAcc), hurts FT (-3.5% CitAcc)
