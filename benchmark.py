"""
Benchmark for evaluating models on Kazakhstan legal domain.
Supports: fine-tuned, baseline (untuned), and RAG evaluation.
Outputs per-domain and per-language breakdowns (Experiments 1 & 4).

Usage:
    python benchmark.py --model lora_qwen3_8b
    python benchmark.py --baseline unsloth/Qwen3-8B-unsloth-bnb-4bit
    python benchmark.py --model lora_qwen3_8b --samples 200 --output results/8b_ft.json
"""
import json
import re
import time
import argparse
import torch
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Set
from collections import defaultdict

# Import metrics BEFORE unsloth (unsloth patches sys.modules and can break imports)
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("WARNING: rouge-score not installed: pip install rouge-score")

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    BLEU_AVAILABLE = True
except ImportError:
    BLEU_AVAILABLE = False
    print("WARNING: nltk not installed: pip install nltk")

try:
    from bert_score import score as bert_score_fn
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False
    print("WARNING: bert-score not installed: pip install bert-score")

from unsloth import FastLanguageModel
from tqdm import tqdm

from config import ALPACA_PROMPT, VAL_FILE


# ============== LANGUAGE & DOMAIN CLASSIFICATION ==============

KZ_CHARS = set('әғқңөүұіӘҒҚҢӨҮҰІ')

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
    if any(c in KZ_CHARS for c in text):
        return 'kz'
    return 'ru'


def classify_domain(text: str) -> str:
    """Return primary domain (first match) or 'other'."""
    text_lower = text.lower()
    for domain, keywords in DOMAIN_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return domain
    return 'other'


# ============== LEGAL CITATION EXTRACTION ==============

def extract_legal_citations(text: str) -> Set[str]:
    """Extract legal citations from text (article numbers, law references)."""
    citations = set()
    text_lower = text.lower()

    for m in re.finditer(r'(?:стать[яиейю]|ст\.?)\s*(\d+)', text_lower):
        citations.add(f"ст.{m.group(1)}")

    for m in re.finditer(r'(?:пункт[а-я]*|п\.?)\s*(\d+)', text_lower):
        citations.add(f"п.{m.group(1)}")

    for m in re.finditer(r'закон[а-я]*\s*(?:№|номер)?\s*(\d[\d\-\.]*\d)', text_lower):
        citations.add(f"закон_{m.group(1)}")

    for m in re.finditer(r'кодекс[а-я]*.*?стать[яиейю]\s*(\d+)', text_lower):
        citations.add(f"ст.{m.group(1)}")

    for m in re.finditer(r'\b(\d+)[\-–](I{1,4}V?|V?I{0,4}|[IVX]+)\b', text):
        citations.add(f"ref_{m.group(0)}")

    return citations


# ============== METRICS ==============

def calculate_rouge(pred: str, reference: str, scorer) -> Dict[str, float]:
    scores = scorer.score(reference, pred)
    return {
        'rouge1': scores['rouge1'].fmeasure,
        'rouge2': scores['rouge2'].fmeasure,
        'rougeL': scores['rougeL'].fmeasure,
    }


def calculate_bleu(pred: str, reference: str) -> float:
    pred_tokens = pred.lower().split()
    ref_tokens = [reference.lower().split()]
    if not pred_tokens or not ref_tokens[0]:
        return 0.0
    smoothing = SmoothingFunction().method1
    try:
        return sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smoothing)
    except Exception:
        return 0.0


def calculate_citation_accuracy(pred: str, reference: str) -> float:
    ref_citations = extract_legal_citations(reference)
    if not ref_citations:
        return 1.0
    pred_citations = extract_legal_citations(pred)
    return len(ref_citations & pred_citations) / len(ref_citations)


def calculate_hallucination_rate(pred: str, reference: str) -> float:
    pred_citations = extract_legal_citations(pred)
    if not pred_citations:
        return 0.0
    ref_citations = extract_legal_citations(reference)
    return len(pred_citations - ref_citations) / len(pred_citations)


def calculate_key_info(pred: str, reference: str) -> float:
    ref_numbers = set(re.findall(r'\d+', reference))
    pred_numbers = set(re.findall(r'\d+', pred))
    number_overlap = len(ref_numbers & pred_numbers) / len(ref_numbers) if ref_numbers else 1.0

    legal_terms = [
        'статья', 'закон', 'кодекс', 'право', 'обязанность', 'договор',
        'срок', 'штраф', 'суд', 'иск', 'заявление', 'документ', 'лицо',
        'гражданин', 'организация', 'государство', 'орган', 'решение'
    ]
    ref_lower = reference.lower()
    pred_lower = pred.lower()
    ref_terms = [t for t in legal_terms if t in ref_lower]
    term_overlap = sum(1 for t in ref_terms if t in pred_lower) / len(ref_terms) if ref_terms else 1.0

    return (number_overlap + term_overlap) / 2


def calculate_length_ratio(pred: str, reference: str) -> float:
    pred_len = len(pred.split())
    ref_len = len(reference.split())
    if ref_len == 0:
        return 0.0
    ratio = pred_len / ref_len
    return 1 / ratio if ratio > 1 else ratio


# ============== MODEL LOADING & INFERENCE ==============

def load_model(model_path: str):
    print(f"Loading model: {model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    print("Model loaded")
    return model, tokenizer


def generate_response(model, tokenizer, instruction: str, input_text: str,
                      max_new_tokens: int = 1024) -> Tuple[str, float]:
    prompt = ALPACA_PROMPT.format(instruction, input_text, "")
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            top_p=0.9,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    gen_time = time.time() - start_time
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip(), gen_time


# ============== BENCHMARK ==============

def compute_bertscore_batch(predictions: List[str], references: List[str]) -> List[float]:
    """BERTScore with xlm-roberta-large for multilingual (RU/KZ) support."""
    P, R, F1 = bert_score_fn(
        predictions, references,
        model_type="xlm-roberta-large",
        verbose=False,
    )
    return F1.tolist()


def run_benchmark(model, tokenizer, test_data: List[Dict],
                  max_new_tokens: int = 1024) -> Dict:
    print("\n" + "=" * 60)
    print("RUNNING BENCHMARK")
    print("=" * 60)

    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False) if ROUGE_AVAILABLE else None

    results = []
    predictions_for_bert = []
    references_for_bert = []
    metrics_sum = defaultdict(float)
    total_gen_time = 0

    for i, sample in enumerate(tqdm(test_data, desc="Evaluating")):
        instruction = sample['instruction']
        input_text = sample.get('input', '')
        reference = sample['output']
        full_text = f"{instruction} {input_text} {reference}"

        # Classify sample
        lang = detect_language(full_text)
        domain = classify_domain(full_text)

        prediction, gen_time = generate_response(model, tokenizer, instruction, input_text, max_new_tokens)
        total_gen_time += gen_time

        metrics = {
            'citation_accuracy': calculate_citation_accuracy(prediction, reference),
            'hallucination_rate': calculate_hallucination_rate(prediction, reference),
            'key_info': calculate_key_info(prediction, reference),
            'length_ratio': calculate_length_ratio(prediction, reference),
            'gen_time': gen_time,
        }

        if ROUGE_AVAILABLE:
            metrics.update(calculate_rouge(prediction, reference, rouge))

        if BLEU_AVAILABLE:
            metrics['bleu'] = calculate_bleu(prediction, reference)

        if BERTSCORE_AVAILABLE:
            predictions_for_bert.append(prediction)
            references_for_bert.append(reference)

        for k, v in metrics.items():
            metrics_sum[k] += v

        results.append({
            'id': i,
            'language': lang,
            'domain': domain,
            'instruction': instruction,
            'input': input_text,
            'reference': reference,
            'prediction': prediction,
            'metrics': metrics,
        })

    # Batch BERTScore
    if BERTSCORE_AVAILABLE and predictions_for_bert:
        print("Computing BERTScore (xlm-roberta-large)...")
        bert_scores = compute_bertscore_batch(predictions_for_bert, references_for_bert)
        for i, score in enumerate(bert_scores):
            results[i]['metrics']['bertscore_f1'] = score
            metrics_sum['bertscore_f1'] += score

    num_samples = len(test_data)
    avg_metrics = {k: v / num_samples for k, v in metrics_sum.items()}

    # Per-domain breakdown
    domain_metrics = compute_breakdown(results, 'domain')
    # Per-language breakdown
    lang_metrics = compute_breakdown(results, 'language')

    # Bootstrap confidence intervals
    print("Computing bootstrap confidence intervals...")
    np.random.seed(42)
    confidence_intervals = compute_bootstrap_ci(results)

    return {
        'num_samples': num_samples,
        'total_time': total_gen_time,
        'avg_time_per_sample': total_gen_time / num_samples,
        'avg_metrics': avg_metrics,
        'confidence_intervals': confidence_intervals,
        'by_domain': domain_metrics,
        'by_language': lang_metrics,
        'results': results,
    }


def compute_breakdown(results: List[Dict], key: str) -> Dict:
    """Compute average metrics grouped by a key (domain or language)."""
    groups = defaultdict(list)
    for r in results:
        groups[r[key]].append(r['metrics'])

    breakdown = {}
    for group_name, metrics_list in sorted(groups.items()):
        n = len(metrics_list)
        avg = defaultdict(float)
        for m in metrics_list:
            for k, v in m.items():
                avg[k] += v
        breakdown[group_name] = {
            'count': n,
            'metrics': {k: v / n for k, v in avg.items()},
        }
    return breakdown


# ============== BOOTSTRAP CONFIDENCE INTERVALS ==============

def compute_bootstrap_ci(results: List[Dict], n_bootstrap: int = 1000,
                         ci: float = 0.95) -> Dict[str, Dict]:
    """Compute bootstrap 95% confidence intervals for all metrics."""
    if not results:
        return {}

    metric_keys = [k for k in results[0]['metrics'].keys() if k != 'gen_time']
    n = len(results)
    alpha = (1 - ci) / 2

    ci_results = {}
    for key in metric_keys:
        values = np.array([r['metrics'].get(key, 0) for r in results])
        boot_means = np.array([
            np.mean(np.random.choice(values, size=n, replace=True))
            for _ in range(n_bootstrap)
        ])
        lower = float(np.percentile(boot_means, alpha * 100))
        upper = float(np.percentile(boot_means, (1 - alpha) * 100))
        mean = float(np.mean(values))
        ci_results[key] = {
            'mean': mean,
            'ci_lower': lower,
            'ci_upper': upper,
            'std': float(np.std(values)),
        }

    return ci_results


# ============== OUTPUT ==============

METRIC_NAMES = [
    ('bertscore_f1', 'BERTScore F1'),
    ('rougeL', 'ROUGE-L'),
    ('rouge1', 'ROUGE-1'),
    ('rouge2', 'ROUGE-2'),
    ('bleu', 'BLEU'),
    ('citation_accuracy', 'Citation Acc'),
    ('hallucination_rate', 'Halluc. Rate'),
    ('key_info', 'Key Info'),
    ('length_ratio', 'Length Ratio'),
    ('gen_time', 'Latency (s)'),
]


def print_results(benchmark_results: Dict, model_name: str, is_baseline: bool):
    avg = benchmark_results['avg_metrics']

    print("\n" + "=" * 60)
    print(f"RESULTS {'(BASELINE)' if is_baseline else '(FINE-TUNED)'}: {model_name}")
    print("=" * 60)
    print(f"Samples: {benchmark_results['num_samples']}, "
          f"Total: {benchmark_results['total_time']:.1f}s, "
          f"Avg latency: {avg.get('gen_time', 0):.2f}s")

    ci = benchmark_results.get('confidence_intervals', {})
    print(f"\n--- Overall Metrics (95% CI) ---")
    for key, name in METRIC_NAMES:
        if key in avg:
            if key == 'gen_time':
                print(f"  {name:<20} {avg[key]:.2f}")
            elif key in ci:
                c = ci[key]
                print(f"  {name:<20} {avg[key]*100:.2f}%  [{c['ci_lower']*100:.2f}%, {c['ci_upper']*100:.2f}%]")
            else:
                print(f"  {name:<20} {avg[key]*100:.2f}%")

    # Domain breakdown
    print(f"\n--- By Legal Domain ---")
    _print_breakdown(benchmark_results['by_domain'])

    # Language breakdown
    print(f"\n--- By Language ---")
    _print_breakdown(benchmark_results['by_language'])

    print("=" * 60)


def _print_breakdown(breakdown: Dict):
    # Header
    cols = ['bertscore_f1', 'rougeL', 'citation_accuracy', 'hallucination_rate', 'key_info']
    header = f"  {'Group':<16} {'N':>4}"
    for c in cols:
        short = c.replace('_', ' ').title()[:10]
        header += f" {short:>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for name, data in breakdown.items():
        m = data['metrics']
        row = f"  {name:<16} {data['count']:>4}"
        for c in cols:
            val = m.get(c, 0)
            row += f" {val*100:>9.1f}%"
        print(row)


def save_results(benchmark_results: Dict, model_name: str,
                 is_baseline: bool, output_file: str = None):
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = model_name.replace("/", "_")
        prefix = "baseline" if is_baseline else "benchmark"
        output_file = f"results/{prefix}_{safe_name}_{timestamp}.json"

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # GPU info
    gpu_info = {}
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        gpu_info = {
            'name': gpu.name,
            'vram_gb': round(gpu.total_memory / 1024**3, 1),
            'peak_vram_gb': round(torch.cuda.max_memory_reserved() / 1024**3, 1),
        }

    save_data = {
        'model': model_name,
        'is_baseline': is_baseline,
        'num_samples': benchmark_results['num_samples'],
        'total_time': benchmark_results['total_time'],
        'avg_metrics': benchmark_results['avg_metrics'],
        'confidence_intervals': benchmark_results.get('confidence_intervals', {}),
        'by_domain': benchmark_results['by_domain'],
        'by_language': benchmark_results['by_language'],
        'gpu': gpu_info,
        'timestamp': datetime.now().isoformat(),
        'detailed_results': benchmark_results['results'],
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved: {output_file}")
    return output_file


def show_examples(benchmark_results: Dict, num_examples: int = 3):
    print("\n--- Sample Predictions ---")
    results = benchmark_results['results']
    examples = random.sample(results, min(num_examples, len(results)))

    for i, ex in enumerate(examples, 1):
        print(f"\n[{i}] [{ex['language'].upper()}] [{ex['domain']}]")
        print(f"  Q: {ex['instruction']}")
        print(f"  Ref: {ex['reference']}")
        print(f"  Pred: {ex['prediction']}")
        m = ex['metrics']
        parts = []
        if 'bertscore_f1' in m:
            parts.append(f"BERT={m['bertscore_f1']*100:.1f}%")
        parts.append(f"CitAcc={m['citation_accuracy']*100:.0f}%")
        parts.append(f"ROUGE-L={m.get('rougeL', 0)*100:.1f}%")
        print(f"  Metrics: {' | '.join(parts)}")


# ============== MAIN ==============

def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark Qwen3 models on Kazakhstan legal QA')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--model', type=str, help='Path to fine-tuned LoRA model')
    group.add_argument('--baseline', type=str, help='Base model name for baseline evaluation')
    parser.add_argument('--samples', type=int, default=100, help='Number of test samples')
    parser.add_argument('--output', type=str, default=None, help='Output JSON path')
    parser.add_argument('--max-tokens', type=int, default=1024, help='Max new tokens')
    return parser.parse_args()


def main():
    args = parse_args()

    is_baseline = args.baseline is not None
    model_path = args.baseline if is_baseline else args.model

    print("=" * 60)
    tag = "BASELINE" if is_baseline else "FINE-TUNED"
    print(f"BENCHMARK ({tag}): {model_path}")
    print(f"Samples: {args.samples}")
    print("=" * 60)

    if not is_baseline and not Path(model_path).exists():
        print(f"ERROR: Model not found: {model_path}")
        return

    model, tokenizer = load_model(model_path)

    # Load test data
    print(f"Loading test data: {VAL_FILE}")
    data = []
    with open(VAL_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    print(f"  Total records: {len(data)}")

    if args.samples and args.samples < len(data):
        random.seed(42)
        data = random.sample(data, args.samples)
        print(f"  Selected: {args.samples}")

    results = run_benchmark(model, tokenizer, data, args.max_tokens)

    print_results(results, model_path, is_baseline)
    show_examples(results)
    output_file = save_results(results, model_path, is_baseline, args.output)

    print("\nBenchmark complete!")
    return output_file


if __name__ == "__main__":
    main()