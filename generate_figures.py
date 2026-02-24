"""
Generate all publication-quality figures for the paper.
Outputs to paper/figures/
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

# === Style setup ===
sns.set_theme(style="whitegrid", font_scale=1.2)
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'DejaVu Sans',
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
})

OUT = Path("paper/figures")
OUT.mkdir(parents=True, exist_ok=True)

COLORS = {
    'ft': '#2196F3',       # Blue
    'base': '#FF9800',     # Orange
    'gpt4o': '#4CAF50',    # Green
    'gpt4o_mini': '#9C27B0', # Purple
    'rag': '#F44336',      # Red
    'accent': '#607D8B',   # Grey
}


# ============================================================
# Fig 1: Training Loss Curve (4B)
# ============================================================
def fig1_training_loss():
    with open("outputs_4b/checkpoint-10653/trainer_state.json") as f:
        state = json.load(f)
    logs = [l for l in state['log_history'] if 'loss' in l]

    steps = [l['step'] for l in logs]
    losses = [l['loss'] for l in logs]
    lrs = [l['learning_rate'] for l in logs]

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Loss
    ax1.plot(steps, losses, color=COLORS['ft'], linewidth=1.5, alpha=0.4, label='Loss (raw)')
    # Smoothed loss (moving average)
    window = 10
    smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
    smoothed_steps = steps[window-1:]
    ax1.plot(smoothed_steps, smoothed, color=COLORS['ft'], linewidth=2.5, label=f'Loss (MA-{window})')

    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Loss', color=COLORS['ft'])
    ax1.tick_params(axis='y', labelcolor=COLORS['ft'])
    ax1.set_ylim(0, max(losses) * 1.05)

    # Learning rate on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(steps, lrs, color=COLORS['accent'], linewidth=1.5, linestyle='--', alpha=0.7, label='Learning Rate')
    ax2.set_ylabel('Learning Rate', color=COLORS['accent'])
    ax2.tick_params(axis='y', labelcolor=COLORS['accent'])
    ax2.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    ax1.set_title('Qwen3-4B LoRA Training: Loss and Learning Rate')
    ax1.axhline(y=smoothed[-1], color=COLORS['ft'], linestyle=':', alpha=0.4)
    ax1.annotate(f'Final loss: {losses[-1]:.2f}',
                xy=(steps[-1], losses[-1]), fontsize=9, color=COLORS['ft'],
                xytext=(-80, 20), textcoords='offset points',
                arrowprops=dict(arrowstyle='->', color=COLORS['ft']))

    plt.savefig(OUT / "fig1_training_loss.png")
    plt.close()
    print("  fig1_training_loss.png")


# ============================================================
# Fig 2: Scaling Analysis (4B vs 8B vs 14B)
# ============================================================
def fig2_scaling():
    models = ['4B', '8B', '14B']
    params = [4, 8, 14]

    # From benchmark results
    ft_bert = [89.6, 89.9, 90.1]
    ft_bert_ci = [0.3, 0.3, 0.3]
    base_bert = [81.0, 81.7, 81.8]
    base_bert_ci = [0.3, 0.2, 0.2]

    ft_cit = [79.7, 78.8, 80.5]
    ft_cit_ci = [3.2, 3.3, 3.0]
    base_cit = [73.5, 73.3, 73.1]
    base_cit_ci = [3.8, 3.8, 3.8]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(len(models))
    w = 0.35

    # BERTScore
    bars1 = ax1.bar(x - w/2, ft_bert, w, yerr=ft_bert_ci, capsize=5,
                    color=COLORS['ft'], label='Fine-tuned', alpha=0.85)
    bars2 = ax1.bar(x + w/2, base_bert, w, yerr=base_bert_ci, capsize=5,
                    color=COLORS['base'], label='Base', alpha=0.85)
    ax1.set_ylabel('BERTScore F1 (%)')
    ax1.set_title('BERTScore F1 by Model Size')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'Qwen3-{m}' for m in models])
    ax1.set_ylim(78, 92)
    ax1.legend()
    ax1.bar_label(bars1, fmt='%.1f', padding=3, fontsize=9)
    ax1.bar_label(bars2, fmt='%.1f', padding=3, fontsize=9)

    # Add GPT-4o reference lines
    ax1.axhline(y=87.2, color=COLORS['gpt4o'], linestyle='--', alpha=0.7, linewidth=1.5)
    ax1.text(-0.4, 87.4, 'GPT-4o (87.2)', color=COLORS['gpt4o'], fontsize=8, ha='left')
    ax1.axhline(y=86.7, color=COLORS['gpt4o_mini'], linestyle=':', alpha=0.7, linewidth=1.5)
    ax1.text(-0.4, 85.9, 'GPT-4o-mini (86.7)', color=COLORS['gpt4o_mini'], fontsize=8, ha='left')

    # Citation Accuracy
    bars3 = ax2.bar(x - w/2, ft_cit, w, yerr=ft_cit_ci, capsize=5,
                    color=COLORS['ft'], label='Fine-tuned', alpha=0.85)
    bars4 = ax2.bar(x + w/2, base_cit, w, yerr=base_cit_ci, capsize=5,
                    color=COLORS['base'], label='Base', alpha=0.85)
    ax2.set_ylabel('Citation Accuracy (%)')
    ax2.set_title('Citation Accuracy by Model Size')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'Qwen3-{m}' for m in models])
    ax2.set_ylim(65, 88)
    ax2.legend()
    ax2.bar_label(bars3, fmt='%.1f', padding=3, fontsize=9)
    ax2.bar_label(bars4, fmt='%.1f', padding=3, fontsize=9)

    ax2.axhline(y=76.3, color=COLORS['gpt4o'], linestyle='--', alpha=0.7, linewidth=1.5)
    ax2.text(-0.4, 76.8, 'GPT-4o (76.3)', color=COLORS['gpt4o'], fontsize=8, ha='left')

    plt.tight_layout()
    plt.savefig(OUT / "fig2_scaling_analysis.png")
    plt.close()
    print("  fig2_scaling_analysis.png")


# ============================================================
# Fig 3: Domain Breakdown (14B FT)
# ============================================================
def fig3_domain():
    # From 14B FT results
    domains = ['Criminal', 'Land', 'Other', 'Admin.', 'Tax', 'Business', 'Const.', 'Family', 'Labor', 'Civil']
    cit_acc = [95.0, 92.3, 94.2, 93.1, 87.4, 83.0, 73.8, 80.0, 59.3, 57.2]
    halluc =  [5.0,  3.8,  9.6,  2.1, 14.6, 31.2, 23.3, 36.0, 62.8, 62.1]
    n_samples = [20, 26, 151, 24, 90, 20, 6, 5, 43, 115]

    # Sort by citation accuracy
    idx = np.argsort(cit_acc)[::-1]
    domains = [domains[i] for i in idx]
    cit_acc = [cit_acc[i] for i in idx]
    halluc = [halluc[i] for i in idx]
    n_samples = [n_samples[i] for i in idx]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(domains))
    w = 0.38

    bars1 = ax.bar(x - w/2, cit_acc, w, color=COLORS['ft'], alpha=0.85, label='Citation Accuracy')
    bars2 = ax.bar(x + w/2, halluc, w, color=COLORS['rag'], alpha=0.75, label='Hallucination Rate')

    ax.set_ylabel('Percentage (%)')
    ax.set_title('Qwen3-14B FT: Citation Accuracy and Hallucination Rate by Legal Domain')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{d}\n(n={n})' for d, n in zip(domains, n_samples)], fontsize=9)
    ax.set_ylim(0, 105)
    ax.legend(loc='upper right')

    ax.bar_label(bars1, fmt='%.0f%%', padding=2, fontsize=8)
    ax.bar_label(bars2, fmt='%.0f%%', padding=2, fontsize=8)

    plt.tight_layout()
    plt.savefig(OUT / "fig3_domain_breakdown.png")
    plt.close()
    print("  fig3_domain_breakdown.png")


# ============================================================
# Fig 4: Language Comparison
# ============================================================
def fig4_language():
    metrics = ['BERTScore F1', 'Citation Acc.', 'Key Info', 'Halluc. Rate']
    kz_vals = [91.8, 95.3, 92.5, 3.7]
    ru_vals = [89.6, 76.1, 67.6, 34.7]

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(metrics))
    w = 0.35

    bars1 = ax.bar(x - w/2, kz_vals, w, color='#26A69A', alpha=0.85, label='Kazakh (n=115)')
    bars2 = ax.bar(x + w/2, ru_vals, w, color='#5C6BC0', alpha=0.85, label='Russian (n=385)')

    ax.set_ylabel('Percentage (%)')
    ax.set_title('Qwen3-14B FT: Performance by Language')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 105)
    ax.legend()

    ax.bar_label(bars1, fmt='%.1f%%', padding=3, fontsize=9)
    ax.bar_label(bars2, fmt='%.1f%%', padding=3, fontsize=9)

    # Add note
    ax.annotate('Lower is better ↓', xy=(3, 40), fontsize=9, color='gray', ha='center')

    plt.tight_layout()
    plt.savefig(OUT / "fig4_language_comparison.png")
    plt.close()
    print("  fig4_language_comparison.png")


# ============================================================
# Fig 5: RAG Ablation
# ============================================================
def fig5_rag():
    configs = ['Only', '+RAG(1)', '+RAG(3)', '+RAG(5)']

    ft_bert = [89.8, 88.3, 88.0, 88.1]
    ft_cit  = [81.9, 77.2, 77.6, 78.4]
    base_bert = [82.6, 82.8, 83.1, 83.2]
    base_cit  = [73.5, 75.6, 78.1, 80.3]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(len(configs))

    # BERTScore
    ax1.plot(x, ft_bert, 'o-', color=COLORS['ft'], linewidth=2.5, markersize=8, label='FT')
    ax1.plot(x, base_bert, 's--', color=COLORS['base'], linewidth=2.5, markersize=8, label='Base')
    ax1.set_ylabel('BERTScore F1 (%)')
    ax1.set_title('BERTScore F1')
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs)
    ax1.set_ylim(80, 92)
    ax1.legend()
    for i, (fv, bv) in enumerate(zip(ft_bert, base_bert)):
        ax1.annotate(f'{fv}', (i, fv), textcoords='offset points', xytext=(0, 10), ha='center', fontsize=9, color=COLORS['ft'])
        ax1.annotate(f'{bv}', (i, bv), textcoords='offset points', xytext=(0, -15), ha='center', fontsize=9, color=COLORS['base'])

    # Fill between to show FT advantage
    ax1.fill_between(x, ft_bert, base_bert, alpha=0.1, color=COLORS['ft'])

    # Citation Accuracy
    ax2.plot(x, ft_cit, 'o-', color=COLORS['ft'], linewidth=2.5, markersize=8, label='FT')
    ax2.plot(x, base_cit, 's--', color=COLORS['base'], linewidth=2.5, markersize=8, label='Base')
    ax2.set_ylabel('Citation Accuracy (%)')
    ax2.set_title('Citation Accuracy')
    ax2.set_xticks(x)
    ax2.set_xticklabels(configs)
    ax2.set_ylim(70, 85)
    ax2.legend()
    for i, (fv, bv) in enumerate(zip(ft_cit, base_cit)):
        ax2.annotate(f'{fv}', (i, fv), textcoords='offset points', xytext=(0, 10), ha='center', fontsize=9, color=COLORS['ft'])
        ax2.annotate(f'{bv}', (i, bv), textcoords='offset points', xytext=(0, -15), ha='center', fontsize=9, color=COLORS['base'])

    ax2.fill_between(x, ft_cit, base_cit, alpha=0.1, color=COLORS['ft'])

    # Arrow annotation
    ax2.annotate('RAG closes gap', xy=(3, 80.3), xytext=(1.5, 83),
                arrowprops=dict(arrowstyle='->', color=COLORS['base']),
                fontsize=9, color=COLORS['base'])

    fig.suptitle('RAG Ablation Study (200 samples)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(OUT / "fig5_rag_ablation.png")
    plt.close()
    print("  fig5_rag_ablation.png")


# ============================================================
# Fig 6: Human Evaluation
# ============================================================
def fig6_human_eval():
    models = ['14B FT', '8B FT', '4B FT', '14B Base', '4B Base', '8B Base']
    correctness = [3.90, 3.70, 3.00, 3.42, 3.42, 3.24]
    completeness = [4.70, 4.66, 3.58, 1.90, 1.96, 1.86]
    relevance = [3.95, 3.73, 3.50, 3.50, 3.48, 3.26]

    colors = [COLORS['ft']]*3 + [COLORS['base']]*3

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, data, title in zip(axes,
                                [correctness, completeness, relevance],
                                ['Correctness', 'Completeness', 'Relevance']):
        bars = ax.barh(range(len(models)), data, color=colors, alpha=0.85, height=0.6)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models)
        ax.set_xlabel('Score (1-5)')
        ax.set_title(title)
        ax.set_xlim(0, 5.3)
        ax.axvline(x=3, color='gray', linestyle=':', alpha=0.5)
        ax.invert_yaxis()

        for bar, val in zip(bars, data):
            ax.text(val + 0.05, bar.get_y() + bar.get_height()/2,
                   f'{val:.2f}', va='center', fontsize=10, fontweight='bold')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=COLORS['ft'], alpha=0.85, label='Fine-tuned'),
                       Patch(facecolor=COLORS['base'], alpha=0.85, label='Base')]
    axes[0].legend(handles=legend_elements, loc='lower right')

    fig.suptitle('Human Evaluation by Legal Experts (50 questions, blind assessment)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(OUT / "fig6_human_eval.png")
    plt.close()
    print("  fig6_human_eval.png")


# ============================================================
# Fig 7: Overall Comparison (Radar/Spider chart)
# ============================================================
def fig7_model_comparison():
    """Grouped bar chart comparing all 8 models on key metrics."""
    models = ['14B FT', '8B FT', '4B FT', 'GPT-4o', 'GPT-4o\nmini', '14B Base', '8B Base', '4B Base']
    bertscore = [90.1, 89.9, 89.6, 87.2, 86.7, 81.8, 81.7, 81.0]
    cit_acc =   [80.5, 78.8, 79.7, 76.3, 75.1, 73.1, 73.3, 73.5]
    halluc =    [27.5, 29.3, 28.1, 83.0, 89.9, 32.5, 23.6, 24.1]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    x = np.arange(len(models))
    bar_colors = [COLORS['ft']]*3 + [COLORS['gpt4o'], COLORS['gpt4o_mini']] + [COLORS['base']]*3

    # BERTScore
    bars = ax1.bar(x, bertscore, color=bar_colors, alpha=0.85, width=0.65)
    ax1.set_ylabel('BERTScore F1 (%)')
    ax1.set_ylim(75, 93)
    ax1.set_title('Automated Metrics: All Models Comparison')
    ax1.bar_label(bars, fmt='%.1f', padding=3, fontsize=9)

    # Citation Accuracy
    bars = ax2.bar(x, cit_acc, color=bar_colors, alpha=0.85, width=0.65)
    ax2.set_ylabel('Citation Acc. (%)')
    ax2.set_ylim(65, 88)
    ax2.bar_label(bars, fmt='%.1f', padding=3, fontsize=9)

    # Hallucination Rate
    bars = ax3.bar(x, halluc, color=bar_colors, alpha=0.85, width=0.65)
    ax3.set_ylabel('Hallucination Rate (%)')
    ax3.set_ylim(0, 100)
    ax3.set_xticks(x)
    ax3.set_xticklabels(models, fontsize=10)
    ax3.bar_label(bars, fmt='%.1f', padding=3, fontsize=9)
    ax3.annotate('↓ Lower is better', xy=(0.02, 0.92), xycoords='axes fraction',
                fontsize=9, color='gray')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['ft'], alpha=0.85, label='Fine-tuned (LoRA)'),
        Patch(facecolor=COLORS['gpt4o'], alpha=0.85, label='GPT-4o'),
        Patch(facecolor=COLORS['gpt4o_mini'], alpha=0.85, label='GPT-4o-mini'),
        Patch(facecolor=COLORS['base'], alpha=0.85, label='Base (pre-trained)'),
    ]
    ax1.legend(handles=legend_elements, loc='lower left', ncol=4)

    plt.tight_layout()
    plt.savefig(OUT / "fig7_model_comparison.png")
    plt.close()
    print("  fig7_model_comparison.png")


# ============================================================
# Fig 8: Latency vs Quality trade-off
# ============================================================
def fig8_latency_quality():
    models =   ['14B FT', '8B FT', '4B FT', 'GPT-4o', 'GPT-4o-mini', '14B Base', '8B Base', '4B Base']
    bert =     [90.1,      89.9,    89.6,    87.2,     86.7,          81.8,       81.7,      81.0]
    latency =  [3.82,      4.59,    3.34,    2.94,     3.86,          19.35,      30.39,     21.01]
    halluc =   [27.5,      29.3,    28.1,    83.0,     89.9,          32.5,       23.6,      24.1]

    fig, ax = plt.subplots(figsize=(10, 7))

    colors = [COLORS['ft']]*3 + [COLORS['gpt4o'], COLORS['gpt4o_mini']] + [COLORS['base']]*3
    sizes = [200 - h*1.5 for h in halluc]  # Larger = less hallucination

    # Per-model label offsets to avoid overlap
    label_offsets = {
        '14B FT':      (8, 8),
        '8B FT':       (8, -14),
        '4B FT':       (-30, 10),
        'GPT-4o':      (8, 8),
        'GPT-4o-mini': (8, -12),
        '14B Base':    (8, 8),
        '8B Base':     (8, 8),
        '4B Base':     (8, -12),
    }
    for i, (m, b, l, h, c, s) in enumerate(zip(models, bert, latency, halluc, colors, sizes)):
        ax.scatter(l, b, s=max(s, 50), c=c, alpha=0.85, edgecolors='white', linewidth=1.5, zorder=5)
        ox, oy = label_offsets.get(m, (8, 8))
        ax.annotate(m, (l, b), textcoords='offset points',
                   xytext=(ox, oy), fontsize=9, fontweight='bold')

    ax.set_xlabel('Inference Latency (seconds)')
    ax.set_ylabel('BERTScore F1 (%)')
    ax.set_title('Quality vs Latency Trade-off (bubble size ∝ lower hallucination)')

    ax.set_xlim(0, 35)
    ax.set_ylim(79, 92)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['ft'], alpha=0.85, label='Fine-tuned'),
        Patch(facecolor=COLORS['gpt4o'], alpha=0.85, label='GPT-4o'),
        Patch(facecolor=COLORS['gpt4o_mini'], alpha=0.85, label='GPT-4o-mini'),
        Patch(facecolor=COLORS['base'], alpha=0.85, label='Base'),
    ]
    ax.legend(handles=legend_elements, loc='lower left')

    plt.tight_layout()
    plt.savefig(OUT / "fig8_latency_quality.png")
    plt.close()
    print("  fig8_latency_quality.png")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("Generating figures...")
    fig1_training_loss()
    fig2_scaling()
    fig3_domain()
    fig4_language()
    fig6_human_eval()
    fig7_model_comparison()
    fig8_latency_quality()
    print(f"\nAll figures saved to {OUT}/")
