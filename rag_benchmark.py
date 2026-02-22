"""
RAG Ablation Benchmark (Experiment 3).
Tests fine-tuned model with different RAG configurations:
  - FT only (no RAG)
  - FT + RAG (top-1, top-3, top-5 similar Q&A pairs)

Uses training dataset as retrieval corpus (56K legal Q&A pairs).

Usage:
    python rag_benchmark.py --model lora_qwen3_4b --corpus combined_data/train.jsonl --samples 200
    python rag_benchmark.py --model lora_qwen3_4b --corpus combined_data/train.jsonl --top-k 1 3 5
"""
import json
import time
import argparse
import torch
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict

# Import metrics BEFORE unsloth
try:
    from bert_score import score as bert_score_fn
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False
    print("WARNING: bert-score not installed: pip install bert-score")

from unsloth import FastLanguageModel
from tqdm import tqdm

from config import ALPACA_PROMPT, VAL_FILE, TRAIN_FILE
from benchmark import (
    calculate_citation_accuracy, calculate_hallucination_rate,
    calculate_key_info, calculate_length_ratio,
    detect_language, classify_domain,
    compute_bootstrap_ci,
)

# ============== RAG PROMPT TEMPLATE ==============

RAG_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context and relevant legal information. Write a response that appropriately completes the request using the provided legal context.

### Instruction:
{}

### Input:
{}

### Legal Context:
{}

### Response:
{}"""


# ============== CORPUS & RETRIEVAL ==============

def load_corpus(corpus_path: str) -> List[Dict]:
    """Load training data as retrieval corpus.

    Supports both formats:
    - Training data: {"instruction": "...", "input": "...", "output": "..."}
    - Custom corpus: {"id": "...", "title": "...", "text": "..."}
    """
    corpus = []
    path = Path(corpus_path)

    files = []
    if path.is_file() and path.suffix == '.jsonl':
        files = [path]
    elif path.is_dir():
        files = sorted(path.glob("*.jsonl"))

    for f in files:
        with open(f, 'r', encoding='utf-8') as fh:
            for idx, line in enumerate(fh):
                if not line.strip():
                    continue
                doc = json.loads(line)
                # Adapt training data format to corpus format
                if 'instruction' in doc and 'output' in doc:
                    question = doc['instruction']
                    inp = doc.get('input', '')
                    corpus.append({
                        'id': idx,
                        'question': f"{question} {inp}".strip(),
                        'text': doc['output'],
                    })
                elif 'text' in doc:
                    corpus.append(doc)

    return corpus


def build_index(corpus: List[Dict]):
    """Build TF-IDF search index over corpus questions."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    # Index by question text (for finding similar questions)
    texts = [doc.get('question', doc.get('text', '')) for doc in corpus]
    vectorizer = TfidfVectorizer(max_features=50000, sublinear_tf=True)
    tfidf_matrix = vectorizer.fit_transform(texts)

    return vectorizer, tfidf_matrix


def retrieve(query: str, vectorizer, tfidf_matrix, corpus: List[Dict],
             top_k: int = 3, exclude_query: str = None) -> List[Dict]:
    """Retrieve top-k relevant Q&A pairs, excluding near-duplicates."""
    from sklearn.metrics.pairwise import cosine_similarity

    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = scores.argsort()[::-1]

    results = []
    for idx in top_indices:
        if len(results) >= top_k:
            break
        # Skip near-exact matches (data leakage prevention)
        if scores[idx] > 0.95:
            continue
        doc = corpus[idx].copy()
        doc['score'] = float(scores[idx])
        results.append(doc)

    return results


def format_context(docs: List[Dict]) -> str:
    """Format retrieved Q&A pairs as context string."""
    parts = []
    for i, doc in enumerate(docs, 1):
        text = doc.get('text', '')[:1000]
        question = doc.get('question', '')[:200]
        parts.append(f"[{i}] Вопрос: {question}\nОтвет: {text}")
    return "\n\n".join(parts)


# ============== BENCHMARK ==============

def run_rag_benchmark(model, tokenizer, test_data: List[Dict],
                      corpus: List[Dict], vectorizer, tfidf_matrix,
                      top_k_values: List[int], max_new_tokens: int = 1024,
                      model_tag: str = "FT") -> Dict:
    """Run RAG ablation with different top-k values. model_tag: 'FT' or 'Base'."""

    configs = [{'name': f'{model_tag} only', 'top_k': 0}]
    for k in top_k_values:
        configs.append({'name': f'{model_tag} + RAG (top-{k})', 'top_k': k})

    all_results = {}

    for config in configs:
        print(f"\n{'='*60}")
        print(f"  {config['name']}")
        print(f"{'='*60}")

        results = []
        predictions_for_bert = []
        references_for_bert = []
        total_time = 0

        for i, sample in enumerate(tqdm(test_data, desc=config['name'])):
            instruction = sample['instruction']
            input_text = sample.get('input', '')
            reference = sample['output']
            query = f"{instruction} {input_text}"

            # Build prompt with or without RAG context
            if config['top_k'] > 0 and vectorizer is not None:
                docs = retrieve(query, vectorizer, tfidf_matrix, corpus,
                               config['top_k'], exclude_query=query)
                context = format_context(docs)
                prompt = RAG_PROMPT.format(instruction, input_text, context, "")
            else:
                prompt = ALPACA_PROMPT.format(instruction, input_text, "")

            inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

            start = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=max_new_tokens,
                    temperature=0.1, top_p=0.9, use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
            gen_time = time.time() - start
            total_time += gen_time

            prediction = tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
            ).strip()

            metrics = {
                'citation_accuracy': calculate_citation_accuracy(prediction, reference),
                'hallucination_rate': calculate_hallucination_rate(prediction, reference),
                'key_info': calculate_key_info(prediction, reference),
                'length_ratio': calculate_length_ratio(prediction, reference),
                'gen_time': gen_time,
            }

            if BERTSCORE_AVAILABLE:
                predictions_for_bert.append(prediction)
                references_for_bert.append(reference)

            full_text = f"{instruction} {input_text} {reference}"
            results.append({
                'id': i,
                'language': detect_language(full_text),
                'domain': classify_domain(full_text),
                'metrics': metrics,
            })

        # Batch BERTScore
        if BERTSCORE_AVAILABLE and predictions_for_bert:
            print("Computing BERTScore (xlm-roberta-large)...")
            P, R, F1 = bert_score_fn(
                predictions_for_bert, references_for_bert,
                model_type="xlm-roberta-large", verbose=False,
            )
            for j, score in enumerate(F1.tolist()):
                results[j]['metrics']['bertscore_f1'] = score

        # Average metrics
        n = len(results)
        avg = {}
        for r in results:
            for k, v in r['metrics'].items():
                avg[k] = avg.get(k, 0) + v / n

        # Bootstrap CI
        print("Computing bootstrap CI...")
        np.random.seed(42)
        ci = compute_bootstrap_ci(results)

        all_results[config['name']] = {
            'config': config,
            'num_samples': n,
            'total_time': total_time,
            'avg_metrics': avg,
            'confidence_intervals': ci,
            'results': results,
        }

        # Print summary
        bert_str = f"BERT={avg.get('bertscore_f1', 0)*100:.1f}%, " if 'bertscore_f1' in avg else ""
        print(f"  {bert_str}CitAcc={avg['citation_accuracy']*100:.1f}%, "
              f"Halluc={avg['hallucination_rate']*100:.1f}%, "
              f"KeyInfo={avg['key_info']*100:.1f}%, "
              f"Lat={avg['gen_time']:.2f}s")

    return all_results


# ============== OUTPUT ==============

def print_rag_table(all_results: Dict):
    """Print RAG ablation comparison table."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 3: RAG ABLATION")
    print("=" * 80)

    metrics = ['bertscore_f1', 'citation_accuracy', 'hallucination_rate', 'key_info']
    short = {
        'bertscore_f1': 'BERT-F1',
        'citation_accuracy': 'Cit.Acc',
        'hallucination_rate': 'Halluc.',
        'key_info': 'KeyInfo',
    }

    header = f"{'Configuration':<25} {'N':>4}"
    for m in metrics:
        header += f" {short[m]:>12}"
    header += f" {'Lat(s)':>7}"
    print(header)
    print("-" * len(header))

    for name, data in all_results.items():
        avg = data['avg_metrics']
        ci = data.get('confidence_intervals', {})
        row = f"{name:<25} {data['num_samples']:>4}"
        for m in metrics:
            val = avg.get(m, 0)
            if m in ci:
                c = ci[m]
                half = (c['ci_upper'] - c['ci_lower']) / 2
                row += f" {val*100:>5.1f}%±{half*100:>4.1f}"
            else:
                row += f" {val*100:>10.1f}%"
        row += f" {avg.get('gen_time', 0):>7.2f}"
        print(row)

    # Delta vs "X only" (first config)
    first_name = list(all_results.keys())[0]
    if first_name in all_results:
        base = all_results[first_name]['avg_metrics']
        print(f"\n--- Delta vs {first_name} ---")
        for name, data in all_results.items():
            if name == first_name:
                continue
            avg = data['avg_metrics']
            parts = [f"{name:<25}"]
            for m in metrics:
                delta = (avg.get(m, 0) - base.get(m, 0)) * 100
                sign = "+" if delta >= 0 else ""
                parts.append(f"{sign}{delta:>5.1f}%")
            print("  ".join(parts))


def main():
    parser = argparse.ArgumentParser(description='RAG Ablation Benchmark (Experiment 3)')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--model', type=str, help='Fine-tuned model path (LoRA adapter)')
    group.add_argument('--baseline', type=str, help='Base model name (e.g. unsloth/Qwen3-4B-unsloth-bnb-4bit)')
    parser.add_argument('--corpus', type=str, required=True, help='Corpus path (training data JSONL)')
    parser.add_argument('--samples', type=int, default=100, help='Number of test samples')
    parser.add_argument('--top-k', nargs='+', type=int, default=[1, 3, 5], help='Top-k values')
    parser.add_argument('--output', type=str, default=None, help='Output JSON path')
    parser.add_argument('--max-tokens', type=int, default=512, help='Max new tokens')
    args = parser.parse_args()

    is_baseline = args.baseline is not None
    model_path = args.baseline if is_baseline else args.model
    tag = "Base" if is_baseline else "FT"

    print("=" * 60)
    print(f"RAG ABLATION BENCHMARK ({tag})")
    print(f"Model: {model_path}")
    print(f"Corpus: {args.corpus}")
    print(f"Samples: {args.samples}")
    print(f"Top-k: {args.top_k}")
    print("=" * 60)

    if not is_baseline and not Path(model_path).exists():
        print(f"ERROR: Model not found: {model_path}")
        return

    # Load corpus (training data)
    print(f"\nLoading corpus: {args.corpus}")
    corpus = load_corpus(args.corpus)
    if not corpus:
        print("ERROR: Empty corpus!")
        return
    print(f"Corpus loaded: {len(corpus)} Q&A pairs")

    # Build TF-IDF index
    print("Building TF-IDF index...")
    vectorizer, tfidf_matrix = build_index(corpus)
    print("Index ready")

    # Load model
    print(f"\nLoading model: {model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path, max_seq_length=2048, dtype=None, load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    # Load test data
    data = []
    with open(VAL_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    if args.samples < len(data):
        random.seed(42)
        data = random.sample(data, args.samples)
    print(f"Test samples: {len(data)}")

    # Run benchmark
    all_results = run_rag_benchmark(
        model, tokenizer, data, corpus, vectorizer, tfidf_matrix,
        args.top_k, args.max_tokens, model_tag=tag
    )

    # Print results table
    print_rag_table(all_results)

    # Save results
    if args.output is None:
        suffix = "base" if is_baseline else "ft"
        args.output = f"results/rag_ablation_{suffix}.json"
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    save_data = {
        'model': model_path,
        'corpus_size': len(corpus),
        'num_samples': args.samples,
        'top_k_values': args.top_k,
        'timestamp': datetime.now().isoformat(),
    }
    for name, data in all_results.items():
        save_data[name] = {
            'config': data['config'],
            'num_samples': data['num_samples'],
            'total_time': data['total_time'],
            'avg_metrics': data['avg_metrics'],
            'confidence_intervals': data.get('confidence_intervals', {}),
        }

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved: {args.output}")


if __name__ == "__main__":
    main()
