"""
Dataset statistics for the paper.
Computes length distributions, language split (RU/KZ), and legal domain categories.

Usage:
    python analyze_dataset.py
"""
import json
import re
from collections import Counter
from pathlib import Path

from config import TRAIN_FILE, VAL_FILE

# Kazakh-specific characters (not present in Russian)
KZ_CHARS = set('әғқңөүұіӘҒҚҢӨҮҰІ')

# Legal domain keywords
DOMAIN_KEYWORDS = {
    'criminal': ['уголовн', 'преступлен', 'наказан', 'қылмыс'],
    'civil': ['гражданск', 'договор', 'сделк', 'азаматтық'],
    'administrative': ['административн', 'штраф', 'правонарушен', 'әкімшілік'],
    'labor': ['трудов', 'работник', 'работодател', 'еңбек'],
    'tax': ['налог', 'салық', 'НДС', 'ҚҚС'],
    'family': ['семейн', 'брак', 'развод', 'отбасы', 'некелес'],
    'land': ['земельн', 'участок', 'жер'],
    'housing': ['жилищн', 'квартир', 'тұрғын'],
    'business': ['предприниматель', 'компани', 'юридическ', 'кәсіпкер'],
    'constitutional': ['конституц', 'основн', 'негізгі'],
}


def detect_language(text: str) -> str:
    """Detect language: 'kz' if Kazakh-specific chars present, else 'ru'."""
    if any(c in KZ_CHARS for c in text):
        return 'kz'
    return 'ru'


def classify_domain(text: str) -> list:
    """Classify legal domain(s) by keyword matching."""
    text_lower = text.lower()
    domains = []
    for domain, keywords in DOMAIN_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            domains.append(domain)
    return domains if domains else ['other']


def compute_stats(values: list) -> dict:
    """Compute basic statistics for a list of numbers."""
    if not values:
        return {'count': 0, 'mean': 0, 'median': 0, 'min': 0, 'max': 0, 'std': 0}
    n = len(values)
    mean = sum(values) / n
    sorted_v = sorted(values)
    median = sorted_v[n // 2] if n % 2 else (sorted_v[n // 2 - 1] + sorted_v[n // 2]) / 2
    variance = sum((x - mean) ** 2 for x in values) / n
    return {
        'count': n,
        'mean': round(mean, 1),
        'median': round(median, 1),
        'min': min(values),
        'max': max(values),
        'std': round(variance ** 0.5, 1),
    }


def load_jsonl(filepath: str) -> list:
    """Load JSONL file."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def analyze(data: list, label: str) -> dict:
    """Analyze a dataset split."""
    instr_lens = []
    input_lens = []
    output_lens = []
    lang_counts = Counter()
    domain_counts = Counter()

    for item in data:
        instr = item.get('instruction', '')
        inp = item.get('input', '')
        out = item.get('output', '')
        full_text = f"{instr} {inp} {out}"

        instr_lens.append(len(instr.split()))
        input_lens.append(len(inp.split()))
        output_lens.append(len(out.split()))

        lang_counts[detect_language(full_text)] += 1

        for domain in classify_domain(full_text):
            domain_counts[domain] += 1

    return {
        'total': len(data),
        'instruction_length': compute_stats(instr_lens),
        'input_length': compute_stats(input_lens),
        'output_length': compute_stats(output_lens),
        'language': dict(lang_counts.most_common()),
        'domains': dict(domain_counts.most_common()),
    }


def main():
    print("=" * 60)
    print("DATASET ANALYSIS")
    print("=" * 60)

    train_data = load_jsonl(TRAIN_FILE)
    val_data = load_jsonl(VAL_FILE)

    print(f"\nTrain file: {TRAIN_FILE} ({len(train_data)} samples)")
    print(f"Val file:   {VAL_FILE} ({len(val_data)} samples)")
    print(f"Total:      {len(train_data) + len(val_data)} samples")

    train_stats = analyze(train_data, "train")
    val_stats = analyze(val_data, "validation")

    # Print results
    for label, stats in [("TRAIN", train_stats), ("VALIDATION", val_stats)]:
        print(f"\n{'=' * 40}")
        print(f"{label} ({stats['total']} samples)")
        print(f"{'=' * 40}")

        print(f"\nLength (words):")
        print(f"  {'Field':<15} {'Mean':>6} {'Median':>7} {'Min':>5} {'Max':>6} {'Std':>6}")
        print(f"  {'-'*45}")
        for field in ['instruction', 'input', 'output']:
            s = stats[f'{field}_length']
            print(f"  {field:<15} {s['mean']:>6} {s['median']:>7} {s['min']:>5} {s['max']:>6} {s['std']:>6}")

        print(f"\nLanguage:")
        for lang, count in stats['language'].items():
            pct = count / stats['total'] * 100
            print(f"  {lang.upper()}: {count} ({pct:.1f}%)")

        print(f"\nLegal domains:")
        for domain, count in stats['domains'].items():
            pct = count / stats['total'] * 100
            print(f"  {domain:<20} {count:>6} ({pct:.1f}%)")

    # Save
    output = {
        'train': train_stats,
        'validation': val_stats,
        'total_samples': len(train_data) + len(val_data),
    }

    output_file = "dataset_stats.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nSaved: {output_file}")


if __name__ == "__main__":
    main()
