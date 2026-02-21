"""
Generate full experiment report: tables, charts, and summary.
Reads all results from results/ and produces report/ directory.
"""
import json
import os
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("WARNING: matplotlib not installed, skipping charts")

RESULTS_DIR = "results"
REPORT_DIR = "report"
FIGURES_DIR = os.path.join(REPORT_DIR, "figures")

os.makedirs(FIGURES_DIR, exist_ok=True)


# ============== LOAD DATA ==============

def load_benchmark(name):
    path = os.path.join(RESULTS_DIR, name)
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


# Load all results
b4b = load_benchmark("baseline_4b.json")
b8b = load_benchmark("baseline_8b.json")
b14b = load_benchmark("baseline_14b.json")
ft4b = load_benchmark("benchmark_4b_ft.json")
ft8b = load_benchmark("benchmark_8b_ft.json")
ft14b = load_benchmark("benchmark_14b_ft.json")
rag_ft = load_benchmark("rag_ablation.json")
rag_base = load_benchmark("rag_ablation_base.json")

# Dataset stats
ds_path = "dataset_stats.json"
ds = json.load(open(ds_path)) if os.path.exists(ds_path) else None


# ============== HELPER ==============

def get_m(d, key):
    return d['avg_metrics'].get(key, 0) * 100

def get_ci(d, key):
    ci = d.get('confidence_intervals', {}).get(key, {})
    if ci:
        return (ci['ci_upper'] - ci['ci_lower']) / 2 * 100
    return 0


# ============== CHART 1: SCALING ANALYSIS ==============

if MATPLOTLIB_AVAILABLE:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    sizes = ['4B', '8B', '14B']
    base_data = [b4b, b8b, b14b]
    ft_data = [ft4b, ft8b, ft14b]

    # BERTScore
    ax = axes[0]
    base_vals = [get_m(d, 'bertscore_f1') for d in base_data]
    ft_vals = [get_m(d, 'bertscore_f1') for d in ft_data]
    base_err = [get_ci(d, 'bertscore_f1') for d in base_data]
    ft_err = [get_ci(d, 'bertscore_f1') for d in ft_data]
    x = np.arange(len(sizes))
    ax.bar(x - 0.2, base_vals, 0.35, yerr=base_err, label='Base', color='#2196F3', capsize=4)
    ax.bar(x + 0.2, ft_vals, 0.35, yerr=ft_err, label='Fine-tuned', color='#4CAF50', capsize=4)
    ax.set_ylabel('BERTScore F1 (%)')
    ax.set_title('BERTScore F1')
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.set_ylim(75, 95)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Citation Accuracy
    ax = axes[1]
    base_vals = [get_m(d, 'citation_accuracy') for d in base_data]
    ft_vals = [get_m(d, 'citation_accuracy') for d in ft_data]
    base_err = [get_ci(d, 'citation_accuracy') for d in base_data]
    ft_err = [get_ci(d, 'citation_accuracy') for d in ft_data]
    ax.bar(x - 0.2, base_vals, 0.35, yerr=base_err, label='Base', color='#2196F3', capsize=4)
    ax.bar(x + 0.2, ft_vals, 0.35, yerr=ft_err, label='Fine-tuned', color='#4CAF50', capsize=4)
    ax.set_ylabel('Citation Accuracy (%)')
    ax.set_title('Citation Accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.set_ylim(60, 90)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Latency
    ax = axes[2]
    base_vals = [d['avg_metrics'].get('gen_time', 0) for d in base_data]
    ft_vals = [d['avg_metrics'].get('gen_time', 0) for d in ft_data]
    ax.bar(x - 0.2, base_vals, 0.35, label='Base', color='#2196F3')
    ax.bar(x + 0.2, ft_vals, 0.35, label='Fine-tuned', color='#4CAF50')
    ax.set_ylabel('Latency (seconds)')
    ax.set_title('Inference Latency')
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Experiment 2: Scaling Analysis (Qwen3 4B / 8B / 14B)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig1_scaling_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: fig1_scaling_analysis.png")


    # ============== CHART 2: RAG ABLATION ==============

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    configs_base = ['Base only', 'Base + RAG (top-1)', 'Base + RAG (top-3)', 'Base + RAG (top-5)']
    configs_ft = ['FT only', 'FT + RAG (top-1)', 'FT + RAG (top-3)', 'FT + RAG (top-5)']
    labels_short = ['No RAG', 'top-1', 'top-3', 'top-5']

    def get_rag_vals(data, configs, metric):
        return [data[c]['avg_metrics'].get(metric, 0) * 100 for c in configs if c in data]

    # BERTScore
    ax = axes[0]
    base_v = get_rag_vals(rag_base, configs_base, 'bertscore_f1')
    ft_v = get_rag_vals(rag_ft, configs_ft, 'bertscore_f1')
    x = np.arange(len(labels_short))
    ax.bar(x - 0.2, base_v, 0.35, label='Base model', color='#FF9800')
    ax.bar(x + 0.2, ft_v, 0.35, label='Fine-tuned', color='#4CAF50')
    ax.set_ylabel('BERTScore F1 (%)')
    ax.set_title('BERTScore F1')
    ax.set_xticks(x)
    ax.set_xticklabels(labels_short)
    ax.set_ylim(78, 92)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Citation Accuracy
    ax = axes[1]
    base_v = get_rag_vals(rag_base, configs_base, 'citation_accuracy')
    ft_v = get_rag_vals(rag_ft, configs_ft, 'citation_accuracy')
    ax.bar(x - 0.2, base_v, 0.35, label='Base model', color='#FF9800')
    ax.bar(x + 0.2, ft_v, 0.35, label='Fine-tuned', color='#4CAF50')
    ax.set_ylabel('Citation Accuracy (%)')
    ax.set_title('Citation Accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels(labels_short)
    ax.set_ylim(65, 90)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Hallucination Rate
    ax = axes[2]
    base_v = get_rag_vals(rag_base, configs_base, 'hallucination_rate')
    ft_v = get_rag_vals(rag_ft, configs_ft, 'hallucination_rate')
    ax.bar(x - 0.2, base_v, 0.35, label='Base model', color='#FF9800')
    ax.bar(x + 0.2, ft_v, 0.35, label='Fine-tuned', color='#4CAF50')
    ax.set_ylabel('Hallucination Rate (%)')
    ax.set_title('Hallucination Rate')
    ax.set_xticks(x)
    ax.set_xticklabels(labels_short)
    ax.set_ylim(15, 45)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Experiment 3: RAG Ablation (Base vs Fine-tuned + RAG)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig2_rag_ablation.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: fig2_rag_ablation.png")


    # ============== CHART 3: DOMAIN BREAKDOWN ==============

    fig, ax = plt.subplots(figsize=(12, 6))

    domains = sorted(ft4b.get('by_domain', {}).keys())
    ft4b_cit = [ft4b['by_domain'][d]['metrics'].get('citation_accuracy', 0) * 100 for d in domains]
    ft8b_cit = [ft8b['by_domain'][d]['metrics'].get('citation_accuracy', 0) * 100 for d in domains]
    ft14b_cit = [ft14b['by_domain'][d]['metrics'].get('citation_accuracy', 0) * 100 for d in domains]

    x = np.arange(len(domains))
    w = 0.25
    ax.bar(x - w, ft4b_cit, w, label='4B FT', color='#4CAF50')
    ax.bar(x, ft8b_cit, w, label='8B FT', color='#2196F3')
    ax.bar(x + w, ft14b_cit, w, label='14B FT', color='#FF5722')
    ax.set_ylabel('Citation Accuracy (%)')
    ax.set_title('Experiment 4a: Citation Accuracy by Legal Domain', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(domains, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig3_domain_breakdown.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: fig3_domain_breakdown.png")


    # ============== CHART 4: LANGUAGE COMPARISON ==============

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for idx, lang in enumerate(['kz', 'ru']):
        ax = axes[idx]
        models = ['4B Base', '4B FT', '8B Base', '8B FT', '14B Base', '14B FT']
        all_data = [b4b, ft4b, b8b, ft8b, b14b, ft14b]
        colors = ['#2196F3', '#4CAF50', '#2196F3', '#4CAF50', '#2196F3', '#4CAF50']

        vals = []
        for d in all_data:
            bl = d.get('by_language', {}).get(lang, {})
            vals.append(bl.get('metrics', {}).get('bertscore_f1', 0) * 100)

        bars = ax.bar(range(len(models)), vals, color=colors)
        ax.set_ylabel('BERTScore F1 (%)')
        ax.set_title(f'Language: {"Kazakh" if lang == "kz" else "Russian"}')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
        ax.set_ylim(75, 95)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Experiment 4b: Performance by Language', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig4_language_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: fig4_language_comparison.png")


    # ============== CHART 5: TRAINING LOSS ==============

    # Try to load training metrics
    import glob
    metrics_files = sorted(glob.glob("logs/metrics_*.json"))
    if metrics_files:
        fig, ax = plt.subplots(figsize=(10, 5))
        for mf in metrics_files:
            with open(mf) as f:
                metrics = json.load(f)
            if 'log_history' in metrics:
                steps = [e['step'] for e in metrics['log_history'] if 'loss' in e]
                losses = [e['loss'] for e in metrics['log_history'] if 'loss' in e]
                size = mf.split('_')[1].upper()
                ax.plot(steps, losses, label=f'{size} model', alpha=0.8)

        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss Curves', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'fig5_training_loss.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("Generated: fig5_training_loss.png")


# ============== GENERATE MARKDOWN REPORT ==============

def fmt(val, mult=100):
    return f"{val * mult:.1f}%"

def fmt_ci(d, key):
    v = d['avg_metrics'].get(key, 0)
    ci = d.get('confidence_intervals', {}).get(key, {})
    if ci:
        hw = (ci['ci_upper'] - ci['ci_lower']) / 2
        return f"{v*100:.1f}% +/- {hw*100:.1f}"
    return f"{v*100:.1f}%"

report = []
report.append("# Experiment Report: Fine-tuning Qwen3 for Kazakhstan Legal Domain\n")
report.append(f"Generated: 2026-02-21\n")

# Dataset
report.append("## 1. Dataset\n")
if ds:
    report.append(f"- **Training samples**: {ds.get('train', {}).get('total', 'N/A')}")
    report.append(f"- **Validation samples**: {ds.get('validation', {}).get('total', 'N/A')}")
    report.append(f"- **Languages**: Russian ({ds.get('train', {}).get('language', {}).get('RU', 'N/A')}), Kazakh ({ds.get('train', {}).get('language', {}).get('KZ', 'N/A')})")
    report.append(f"- **Legal domains**: 11 categories (civil, criminal, tax, labor, etc.)")
else:
    report.append("- 56,802 training / 6,312 validation samples")
    report.append("- Languages: Russian (76%), Kazakh (24%)")
    report.append("- 11 legal domains")
report.append("")

# Experiment 1 & 2: Scaling
report.append("## 2. Experiment 1 & 2: Scaling Analysis (500 samples, 95% CI)\n")
report.append("| Model | BERTScore F1 | Citation Acc | Halluc Rate | Key Info | Latency | VRAM |")
report.append("|-------|-------------|-------------|-------------|----------|---------|------|")
for label, d in [("4B Base", b4b), ("4B FT", ft4b), ("8B Base", b8b), ("8B FT", ft8b), ("14B Base", b14b), ("14B FT", ft14b)]:
    gpu = d.get('gpu', {})
    vram = f"{gpu.get('peak_vram_gb', '?')}G" if gpu else "—"
    report.append(f"| {label} | {fmt_ci(d, 'bertscore_f1')} | {fmt_ci(d, 'citation_accuracy')} | {fmt_ci(d, 'hallucination_rate')} | {fmt_ci(d, 'key_info')} | {d['avg_metrics'].get('gen_time', 0):.1f}s | {vram} |")
report.append("")
report.append("**Key finding**: All FT model sizes show overlapping CIs — 4B is optimal (same accuracy, 2x less VRAM, fastest latency).\n")
report.append("![Scaling Analysis](figures/fig1_scaling_analysis.png)\n")

# Experiment 3: RAG
report.append("## 3. Experiment 3: RAG Ablation (200 samples)\n")
report.append("### 3a. Base + RAG (Can RAG replace Fine-tuning?)\n")
report.append("| Configuration | BERTScore F1 | Citation Acc | Halluc Rate | Key Info | Latency |")
report.append("|---------------|-------------|-------------|-------------|----------|---------|")
for name in ['Base only', 'Base + RAG (top-1)', 'Base + RAG (top-3)', 'Base + RAG (top-5)']:
    if name in rag_base:
        m = rag_base[name]['avg_metrics']
        report.append(f"| {name} | {m.get('bertscore_f1',0)*100:.1f}% | {m['citation_accuracy']*100:.1f}% | {m['hallucination_rate']*100:.1f}% | {m['key_info']*100:.1f}% | {m['gen_time']:.1f}s |")
report.append("")

report.append("### 3b. FT + RAG (Does RAG improve Fine-tuned model?)\n")
report.append("| Configuration | BERTScore F1 | Citation Acc | Halluc Rate | Key Info | Latency |")
report.append("|---------------|-------------|-------------|-------------|----------|---------|")
for name in ['FT only', 'FT + RAG (top-1)', 'FT + RAG (top-3)', 'FT + RAG (top-5)']:
    if name in rag_ft:
        m = rag_ft[name]['avg_metrics']
        report.append(f"| {name} | {m.get('bertscore_f1',0)*100:.1f}% | {m['citation_accuracy']*100:.1f}% | {m['hallucination_rate']*100:.1f}% | {m['key_info']*100:.1f}% | {m['gen_time']:.1f}s |")
report.append("")
report.append("**Key findings**:")
report.append("- RAG improves Base model significantly (+6.8% Citation Accuracy)")
report.append("- RAG does NOT improve FT model — fine-tuning already internalized training data knowledge")
report.append("- FT alone > Base + RAG: BERTScore +6.6%, Latency 4.5x faster\n")
report.append("![RAG Ablation](figures/fig2_rag_ablation.png)\n")

# Experiment 4: Domain/Language
report.append("## 4. Experiment 4: Domain & Language Breakdown\n")
report.append("### 4a. By Legal Domain (FT models)\n")
report.append("| Domain | N | 4B FT CitAcc | 8B FT CitAcc | 14B FT CitAcc |")
report.append("|--------|---|-------------|-------------|--------------|")
for d in sorted(ft4b.get('by_domain', {}).keys()):
    n = ft4b['by_domain'][d]['count']
    v4 = ft4b['by_domain'][d]['metrics'].get('citation_accuracy', 0) * 100
    v8 = ft8b['by_domain'][d]['metrics'].get('citation_accuracy', 0) * 100
    v14 = ft14b['by_domain'][d]['metrics'].get('citation_accuracy', 0) * 100
    report.append(f"| {d} | {n} | {v4:.1f}% | {v8:.1f}% | {v14:.1f}% |")
report.append("")

report.append("### 4b. By Language\n")
report.append("| Model | KZ BERTScore | KZ CitAcc | RU BERTScore | RU CitAcc |")
report.append("|-------|-------------|-----------|-------------|-----------|")
for label, d in [("4B Base", b4b), ("4B FT", ft4b), ("8B FT", ft8b), ("14B FT", ft14b)]:
    kz = d.get('by_language', {}).get('kz', {}).get('metrics', {})
    ru = d.get('by_language', {}).get('ru', {}).get('metrics', {})
    report.append(f"| {label} | {kz.get('bertscore_f1',0)*100:.1f}% | {kz.get('citation_accuracy',0)*100:.1f}% | {ru.get('bertscore_f1',0)*100:.1f}% | {ru.get('citation_accuracy',0)*100:.1f}% |")
report.append("")
report.append("![Domain Breakdown](figures/fig3_domain_breakdown.png)\n")
report.append("![Language Comparison](figures/fig4_language_comparison.png)\n")

# Experiment 5
report.append("## 5. Experiment 5: Human Evaluation\n")
report.append("- **Status**: Evaluation sheets prepared (50 questions x 4 models)")
report.append("- **Models evaluated**: Base 4B, FT 4B, FT 8B, FT 14B")
report.append("- **Criteria**: Correctness (1-5), Completeness (1-5), Relevance (1-5), Hallucination (yes/no)")
report.append("- **Files**: `human_eval/eval_sheet_*.csv` + `human_eval/ИНСТРУКЦИЯ.txt`")
report.append("- Awaiting lawyer evaluations\n")

# Training details
report.append("## 6. Training Details\n")
report.append("| Parameter | Value |")
report.append("|-----------|-------|")
report.append("| Base model | Qwen3 (4B / 8B / 14B) |")
report.append("| Quantization | 4-bit (Unsloth BnB) |")
report.append("| LoRA rank | r=16, alpha=16 |")
report.append("| Target modules | q,k,v,o,gate,up,down proj |")
report.append("| Optimizer | AdamW 8-bit |")
report.append("| LR scheduler | Cosine |")
report.append("| Learning rate | 2e-4 |")
report.append("| Epochs | 3 |")
report.append("| Training data | 56,802 Q&A pairs |")
report.append("| Max seq length | 2048 |")
report.append(f"| 4B training time | ~15.5 hours (RTX 5080 16GB) |")
report.append(f"| 14B training time | ~14.1 hours (RTX 5090 32GB) |")
report.append("")

# Conclusions
report.append("## 7. Key Conclusions\n")
report.append("1. **Fine-tuning is highly effective**: +7-8% BERTScore, +7% Citation Accuracy across all sizes")
report.append("2. **Model size doesn't matter for this domain**: 4B = 8B = 14B after FT (CIs overlap)")
report.append("3. **4B is optimal**: Same quality, 2x less VRAM (7.4G), fastest inference (2.5s)")
report.append("4. **RAG improves base but can't replace FT**: Base+RAG top-5 approaches FT on Citation Acc (80.3% vs 81.9%) but lags on BERTScore (83.2% vs 89.8%)")
report.append("5. **RAG doesn't help FT model**: Fine-tuning already internalized training knowledge")
report.append("6. **Kazakh performs better than Russian**: Higher Citation Acc (95%+ vs 75%) due to simpler legal queries in KZ subset")
report.append("")

# Write report
report_path = os.path.join(REPORT_DIR, "experiment_report.md")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report))
print(f"\nReport written: {report_path}")
print(f"Figures: {FIGURES_DIR}/")
