# Материалы для статьи: Fine-tuning Qwen3 для казахстанского правового домена

## Содержание папки

```
report/
├── README.md                          ← Этот файл (оглавление)
├── experiment_report.md               ← Сводка результатов + ссылки на графики
├── experiment_methodology.md          ← Подробный ход экспериментов
│
├── figures/                           ← Графики для статьи (5 PNG)
│   ├── fig1_scaling_analysis.png      — Exp 2: Base vs FT (4B/8B/14B)
│   ├── fig2_rag_ablation.png          — Exp 3: RAG ablation (Base+RAG vs FT+RAG)
│   ├── fig3_domain_breakdown.png      — Exp 4a: Citation Acc по правовым доменам
│   ├── fig4_language_comparison.png   — Exp 4b: BERTScore по языкам (RU vs KZ)
│   └── fig5_training_loss.png         — Training loss curve (4B)
│
├── data/                              ← Сырые данные экспериментов (JSON)
│   ├── baseline_4b.json               — Base 4B (500 samples, CI)
│   ├── baseline_8b.json               — Base 8B (500 samples, CI)
│   ├── baseline_14b.json              — Base 14B (500 samples, CI)
│   ├── benchmark_4b_ft.json           — FT 4B (500 samples, CI)
│   ├── benchmark_8b_ft.json           — FT 8B (500 samples, CI)
│   ├── benchmark_14b_ft.json          — FT 14B (500 samples, CI)
│   ├── rag_ablation.json              — FT + RAG (200 samples, top-1/3/5)
│   ├── rag_ablation_base.json         — Base + RAG (200 samples, top-1/3/5)
│   ├── dataset_stats.json             — Статистика датасета
│   └── paper_tables.md                — Сводные таблицы (markdown)
│
└── human_eval/                        ← Экспертная оценка юристами
    ├── eval_sheet_*.csv               — Таблица для юристов (50 вопросов × 4 модели)
    ├── model_key_*.json               — Секретный ключ (модель → ответ)
    └── ИНСТРУКЦИЯ.txt                 — Инструкция для экспертов
```

## Как использовать для статьи

### Таблицы
- **Table 1** (Scaling): `experiment_report.md` → Section 2
- **Table 2** (RAG): `experiment_report.md` → Section 3
- **Table 3** (Domains): `experiment_report.md` → Section 4a
- **Table 4** (Languages): `experiment_report.md` → Section 4b

### Графики
- **Figure 1**: `figures/fig1_scaling_analysis.png` — основной результат
- **Figure 2**: `figures/fig2_rag_ablation.png` — RAG vs FT
- **Figure 3**: `figures/fig3_domain_breakdown.png` — по доменам
- **Figure 4**: `figures/fig4_language_comparison.png` — RU vs KZ
- **Figure 5**: `figures/fig5_training_loss.png` — loss curve

### Методология
- Детальное описание: `experiment_methodology.md`
- Гиперпараметры: Section 2.2
- Метрики: Section 3
- Ограничения: Section 10

### Ключевые числа для Abstract
- Dataset: 56,802 train / 6,312 val, RU (76%) + KZ (24%), 11 legal domains
- Best model: Qwen3-4B FT — BERTScore 89.7%, Citation Acc 80.1%, Latency 2.5s
- FT improvement: +7.2% BERTScore, +6.9% Citation Accuracy
- Scaling: 4B = 8B = 14B (CIs overlap) → 4B optimal (2x less VRAM)
- RAG: Base+RAG top-5 CitAcc 80.3% vs FT 81.9% — FT still better, RAG can't replace FT
- RAG doesn't improve FT model — fine-tuning internalizes training knowledge
