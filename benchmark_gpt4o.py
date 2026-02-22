"""
Benchmark GPT-4o on the same validation samples for comparison.
Uses OpenAI API to get responses and computes the same metrics.

Usage:
    python benchmark_gpt4o.py --samples 200 --output results/benchmark_gpt4o.json
"""
import json
import os
import re
import time
import random
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI

from config import VAL_FILE, ALPACA_PROMPT
from benchmark import (
    calculate_citation_accuracy, calculate_hallucination_rate,
    calculate_key_info, calculate_length_ratio,
    detect_language, classify_domain,
    compute_bootstrap_ci,
)

try:
    from bert_score import score as bert_score_fn
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False


def query_gpt4o(client, instruction, input_text, model="gpt-4o-mini"):
    """Send a query to GPT-4o and return response."""
    prompt = f"""Ты — юридический консультант по законодательству Республики Казахстан.
Ответь на вопрос кратко и точно, ссылаясь на конкретные статьи законов РК где это уместно.
Отвечай на том же языке, на котором задан вопрос.

Вопрос: {instruction}
{('Контекст: ' + input_text) if input_text else ''}

Ответ:"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.1,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  API error: {e}")
        return f"[ERROR: {e}]"


def main():
    parser = argparse.ArgumentParser(description='Benchmark GPT-4o on Kazakhstan legal QA')
    parser.add_argument('--samples', type=int, default=200, help='Number of test samples')
    parser.add_argument('--model', type=str, default='gpt-4o-mini', help='OpenAI model name')
    parser.add_argument('--output', type=str, default='results/benchmark_gpt4o.json')
    args = parser.parse_args()

    api_key = os.getenv("OPENAIAPI_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Set OPENAIAPI_KEY in .env file")
        return

    client = OpenAI(api_key=api_key)

    print("=" * 60)
    print(f"GPT-4o BENCHMARK")
    print(f"Model: {args.model}")
    print(f"Samples: {args.samples}")
    print("=" * 60)

    # Load test data (same seed as other benchmarks)
    data = []
    with open(VAL_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    if args.samples < len(data):
        random.seed(42)
        data = random.sample(data, args.samples)
    print(f"Test samples: {len(data)}")

    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False) if ROUGE_AVAILABLE else None

    results = []
    predictions_for_bert = []
    references_for_bert = []
    metrics_sum = defaultdict(float)

    for i, sample in enumerate(data):
        instruction = sample['instruction']
        input_text = sample.get('input', '')
        reference = sample['output']
        full_text = f"{instruction} {input_text} {reference}"

        lang = detect_language(full_text)
        domain = classify_domain(full_text)

        start = time.time()
        prediction = query_gpt4o(client, instruction, input_text, args.model)
        gen_time = time.time() - start

        metrics = {
            'citation_accuracy': calculate_citation_accuracy(prediction, reference),
            'hallucination_rate': calculate_hallucination_rate(prediction, reference),
            'key_info': calculate_key_info(prediction, reference),
            'length_ratio': calculate_length_ratio(prediction, reference),
            'gen_time': gen_time,
        }

        if ROUGE_AVAILABLE:
            scores = rouge.score(reference, prediction)
            metrics['rouge1'] = scores['rouge1'].fmeasure
            metrics['rouge2'] = scores['rouge2'].fmeasure
            metrics['rougeL'] = scores['rougeL'].fmeasure

        if BERTSCORE_AVAILABLE:
            predictions_for_bert.append(prediction)
            references_for_bert.append(reference)

        for k, v in metrics.items():
            metrics_sum[k] += v

        results.append({
            'id': i,
            'language': lang,
            'domain': domain,
            'instruction': instruction[:100],
            'prediction': prediction[:300],
            'reference': reference[:200],
            'metrics': metrics,
        })

        if (i + 1) % 10 == 0:
            avg_cit = metrics_sum['citation_accuracy'] / (i + 1) * 100
            avg_hal = metrics_sum['hallucination_rate'] / (i + 1) * 100
            print(f"  [{i+1}/{len(data)}] CitAcc={avg_cit:.1f}% Halluc={avg_hal:.1f}% Lat={gen_time:.1f}s")

    # Batch BERTScore
    if BERTSCORE_AVAILABLE and predictions_for_bert:
        print("Computing BERTScore (xlm-roberta-large)...")
        P, R, F1 = bert_score_fn(
            predictions_for_bert, references_for_bert,
            model_type="xlm-roberta-large", verbose=False,
        )
        for j, score in enumerate(F1.tolist()):
            results[j]['metrics']['bertscore_f1'] = score
            metrics_sum['bertscore_f1'] += score

    n = len(results)
    avg_metrics = {k: v / n for k, v in metrics_sum.items()}

    # Bootstrap CI
    print("Computing bootstrap CI...")
    np.random.seed(42)
    ci = compute_bootstrap_ci(results)

    # Domain/Language breakdown
    by_domain = defaultdict(list)
    by_language = defaultdict(list)
    for r in results:
        by_domain[r['domain']].append(r['metrics'])
        by_language[r['language']].append(r['metrics'])

    domain_breakdown = {}
    for d, mlist in by_domain.items():
        avg = {k: sum(m.get(k, 0) for m in mlist) / len(mlist) for k in mlist[0]}
        domain_breakdown[d] = {'count': len(mlist), 'metrics': avg}

    lang_breakdown = {}
    for l, mlist in by_language.items():
        avg = {k: sum(m.get(k, 0) for m in mlist) / len(mlist) for k in mlist[0]}
        lang_breakdown[l] = {'count': len(mlist), 'metrics': avg}

    # Print results
    print(f"\n{'='*60}")
    print(f"RESULTS: {args.model}")
    print(f"{'='*60}")
    print(f"Samples: {n}")
    for key in ['bertscore_f1', 'rougeL', 'citation_accuracy', 'hallucination_rate', 'key_info']:
        if key in avg_metrics:
            val = avg_metrics[key]
            if key in ci:
                c = ci[key]
                hw = (c['ci_upper'] - c['ci_lower']) / 2
                print(f"  {key:<25} {val*100:.1f}% +/- {hw*100:.1f}")
            else:
                print(f"  {key:<25} {val*100:.1f}%")
    print(f"  {'latency':<25} {avg_metrics.get('gen_time', 0):.2f}s")

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_data = {
        'model': args.model,
        'is_baseline': True,
        'num_samples': n,
        'total_time': sum(r['metrics']['gen_time'] for r in results),
        'avg_metrics': avg_metrics,
        'confidence_intervals': ci,
        'by_domain': domain_breakdown,
        'by_language': lang_breakdown,
        'timestamp': datetime.now().isoformat(),
        'detailed_results': results,
    }
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved: {args.output}")


if __name__ == "__main__":
    main()
