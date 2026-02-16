"""
Compare benchmark results and generate paper-ready tables.
Experiment 2: Scaling Analysis — Base vs FT (vs FT+RAG) across model sizes.
Experiment 4: Breakdown by domain and language.

Usage:
    python compare_results.py results/
    python compare_results.py results/baseline_4b.json results/benchmark_4b.json results/benchmark_8b.json
"""
import json
import argparse
from pathlib import Path
from typing import List, Dict


def load_results(paths: List[str]) -> List[Dict]:
    """Load benchmark result JSON files."""
    results = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            for f in sorted(path.glob("*.json")):
                with open(f, 'r', encoding='utf-8') as fh:
                    data = json.load(fh)
                    data['_file'] = str(f)
                    results.append(data)
        else:
            with open(path, 'r', encoding='utf-8') as fh:
                data = json.load(fh)
                data['_file'] = str(path)
                results.append(data)
    return results


def model_label(result: Dict) -> str:
    """Human-readable label for a result."""
    model = result['model']
    tag = "Base" if result.get('is_baseline') else "FT"
    # Extract size from model name
    for size in ['4b', '8b', '14b', '32b', '4B', '8B', '14B', '32B']:
        if size.lower() in model.lower():
            return f"Qwen3-{size.upper()} {tag}"
    return f"{model} ({tag})"


# ============== EXPERIMENT 2: SCALING ANALYSIS ==============

def print_scaling_table(results: List[Dict]):
    """Print Experiment 2: Scaling Analysis table."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: SCALING ANALYSIS")
    print("=" * 80)

    # Key metrics for the table
    key_metrics = ['bertscore_f1', 'rougeL', 'citation_accuracy', 'hallucination_rate', 'key_info']
    metric_short = {
        'bertscore_f1': 'BERT-F1',
        'rougeL': 'ROUGE-L',
        'citation_accuracy': 'Cit.Acc',
        'hallucination_rate': 'Halluc.',
        'key_info': 'KeyInfo',
    }

    # Header
    header = f"{'Model':<25} {'N':>4}"
    for m in key_metrics:
        header += f" {metric_short[m]:>8}"
    header += f" {'Lat(s)':>7}"
    if results and 'gpu' in results[0]:
        header += f" {'VRAM':>6}"
    print(header)
    print("-" * len(header))

    for r in sorted(results, key=lambda x: (x.get('is_baseline', False), model_label(x))):
        avg = r['avg_metrics']
        label = model_label(r)
        row = f"{label:<25} {r['num_samples']:>4}"
        for m in key_metrics:
            val = avg.get(m, 0)
            row += f" {val*100:>7.1f}%"
        row += f" {avg.get('gen_time', 0):>7.2f}"
        gpu = r.get('gpu', {})
        if gpu:
            row += f" {gpu.get('peak_vram_gb', '?'):>5}G"
        print(row)

    # Delta table (FT improvement over baseline)
    baselines = {_extract_size(r['model']): r for r in results if r.get('is_baseline')}
    finetuned = {_extract_size(r['model']): r for r in results if not r.get('is_baseline')}

    if baselines and finetuned:
        print(f"\n--- Improvement (FT - Base) ---")
        header = f"{'Size':<10}"
        for m in key_metrics:
            header += f" {metric_short[m]:>8}"
        print(header)
        print("-" * len(header))

        for size in sorted(set(baselines.keys()) & set(finetuned.keys())):
            base_avg = baselines[size]['avg_metrics']
            ft_avg = finetuned[size]['avg_metrics']
            row = f"{size.upper():<10}"
            for m in key_metrics:
                delta = (ft_avg.get(m, 0) - base_avg.get(m, 0)) * 100
                sign = "+" if delta >= 0 else ""
                row += f" {sign}{delta:>6.1f}%"
            print(row)


def _extract_size(model_name: str) -> str:
    """Extract model size (4b, 8b, 14b) from model name."""
    for size in ['4b', '8b', '14b', '32b']:
        if size in model_name.lower():
            return size
    return model_name


# ============== EXPERIMENT 4: DOMAIN & LANGUAGE BREAKDOWN ==============

def print_domain_breakdown(results: List[Dict]):
    """Print Experiment 4: Breakdown by legal domain."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 4a: BREAKDOWN BY LEGAL DOMAIN")
    print("=" * 80)

    key_metrics = ['bertscore_f1', 'rougeL', 'citation_accuracy']
    metric_short = {'bertscore_f1': 'BERT-F1', 'rougeL': 'ROUGE-L', 'citation_accuracy': 'Cit.Acc'}

    for r in results:
        if r.get('is_baseline'):
            continue
        by_domain = r.get('by_domain', {})
        if not by_domain:
            continue

        print(f"\n{model_label(r)}:")
        header = f"  {'Domain':<16} {'N':>4}"
        for m in key_metrics:
            header += f" {metric_short[m]:>8}"
        print(header)
        print("  " + "-" * (len(header) - 2))

        for domain in sorted(by_domain.keys()):
            data = by_domain[domain]
            dm = data['metrics']
            row = f"  {domain:<16} {data['count']:>4}"
            for m in key_metrics:
                row += f" {dm.get(m, 0)*100:>7.1f}%"
            print(row)


def print_language_breakdown(results: List[Dict]):
    """Print Experiment 4: Breakdown by language."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 4b: BREAKDOWN BY LANGUAGE (RU vs KZ)")
    print("=" * 80)

    key_metrics = ['bertscore_f1', 'rougeL', 'citation_accuracy', 'hallucination_rate']
    metric_short = {
        'bertscore_f1': 'BERT-F1', 'rougeL': 'ROUGE-L',
        'citation_accuracy': 'Cit.Acc', 'hallucination_rate': 'Halluc.',
    }

    # Cross-model comparison per language
    header = f"{'Model':<25} {'Lang':>4} {'N':>4}"
    for m in key_metrics:
        header += f" {metric_short[m]:>8}"
    print(header)
    print("-" * len(header))

    for r in sorted(results, key=lambda x: model_label(x)):
        by_lang = r.get('by_language', {})
        if not by_lang:
            continue
        for lang in sorted(by_lang.keys()):
            data = by_lang[lang]
            lm = data['metrics']
            row = f"{model_label(r):<25} {lang.upper():>4} {data['count']:>4}"
            for m in key_metrics:
                row += f" {lm.get(m, 0)*100:>7.1f}%"
            print(row)


# ============== MARKDOWN EXPORT ==============

def export_markdown(results: List[Dict], output_file: str):
    """Export comparison tables as markdown for the paper."""
    lines = []
    lines.append("# Experiment Results\n")

    # Scaling table
    key_metrics = ['bertscore_f1', 'rougeL', 'citation_accuracy', 'hallucination_rate', 'key_info']
    metric_short = {
        'bertscore_f1': 'BERTScore F1',
        'rougeL': 'ROUGE-L',
        'citation_accuracy': 'Citation Acc.',
        'hallucination_rate': 'Halluc. Rate',
        'key_info': 'Key Info',
    }

    lines.append("## Table 1: Model Comparison\n")
    header = "| Model | " + " | ".join(metric_short[m] for m in key_metrics) + " | Latency (s) |"
    sep = "|" + "|".join("---" for _ in range(len(key_metrics) + 2)) + "|"
    lines.append(header)
    lines.append(sep)

    for r in sorted(results, key=lambda x: (x.get('is_baseline', False), model_label(x))):
        avg = r['avg_metrics']
        label = model_label(r)
        cols = [label]
        for m in key_metrics:
            cols.append(f"{avg.get(m, 0)*100:.1f}%")
        cols.append(f"{avg.get('gen_time', 0):.2f}")
        lines.append("| " + " | ".join(cols) + " |")

    lines.append("")

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print(f"\nMarkdown exported: {output_file}")


# ============== MAIN ==============

def main():
    parser = argparse.ArgumentParser(description='Compare benchmark results and generate paper tables')
    parser.add_argument('paths', nargs='+', help='Result JSON files or directories')
    parser.add_argument('--markdown', type=str, default=None, help='Export markdown table to file')
    args = parser.parse_args()

    results = load_results(args.paths)
    if not results:
        print("No result files found!")
        return

    print(f"Loaded {len(results)} result files:")
    for r in results:
        print(f"  {r['_file']}: {model_label(r)} ({r['num_samples']} samples)")

    print_scaling_table(results)
    print_domain_breakdown(results)
    print_language_breakdown(results)

    if args.markdown:
        export_markdown(results, args.markdown)


if __name__ == "__main__":
    main()
