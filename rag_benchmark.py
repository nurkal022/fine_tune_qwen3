"""
RAG Ablation Benchmark (Experiment 3).
Tests fine-tuned model with different RAG configurations:
  - FT only (no RAG)
  - FT + RAG (top-1, top-3, top-5 articles)
  - FT + RAG (oracle — correct article injected)

Requires: a legal document corpus indexed for retrieval.

Usage:
    python rag_benchmark.py --model lora_qwen3_8b --corpus legal_docs/ --samples 100
    python rag_benchmark.py --model lora_qwen3_8b --corpus legal_docs/ --top-k 1 3 5
"""
import json
import time
import argparse
import torch
import random
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple

from config import ALPACA_PROMPT, VAL_FILE

# ============== RAG PROMPT TEMPLATE ==============

RAG_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context and relevant legal articles. Write a response that appropriately completes the request using the provided legal context.

### Instruction:
{}

### Input:
{}

### Legal Context:
{}

### Response:
{}"""


# ============== RETRIEVAL ==============

def load_corpus(corpus_path: str) -> List[Dict]:
    """Load legal document corpus.

    Expected format: directory of JSONL files or single JSONL file.
    Each document: {"id": "...", "title": "...", "text": "...", "article": "ст. 28 ТК РК"}

    TODO: Replace with your actual legal document corpus.
    """
    corpus = []
    path = Path(corpus_path)

    if path.is_file() and path.suffix == '.jsonl':
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    corpus.append(json.loads(line))
    elif path.is_dir():
        for f in sorted(path.glob("*.jsonl")):
            with open(f, 'r', encoding='utf-8') as fh:
                for line in fh:
                    if line.strip():
                        corpus.append(json.loads(line))
    else:
        print(f"WARNING: Corpus not found: {corpus_path}")

    return corpus


def build_index(corpus: List[Dict]):
    """Build simple TF-IDF search index.

    For production, replace with:
    - sentence-transformers + FAISS for dense retrieval
    - Or BM25 via rank_bm25 library
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        print("ERROR: scikit-learn required: pip install scikit-learn")
        return None, None

    texts = [doc.get('text', '') for doc in corpus]
    vectorizer = TfidfVectorizer(max_features=50000)
    tfidf_matrix = vectorizer.fit_transform(texts)

    return vectorizer, tfidf_matrix


def retrieve(query: str, vectorizer, tfidf_matrix, corpus: List[Dict], top_k: int = 3) -> List[Dict]:
    """Retrieve top-k relevant documents for a query."""
    from sklearn.metrics.pairwise import cosine_similarity

    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = scores.argsort()[-top_k:][::-1]

    results = []
    for idx in top_indices:
        doc = corpus[idx].copy()
        doc['score'] = float(scores[idx])
        results.append(doc)
    return results


def format_context(docs: List[Dict]) -> str:
    """Format retrieved documents as context string."""
    parts = []
    for i, doc in enumerate(docs, 1):
        title = doc.get('title', doc.get('article', f'Document {i}'))
        text = doc.get('text', '')[:500]  # Truncate long documents
        parts.append(f"[{i}] {title}\n{text}")
    return "\n\n".join(parts)


# ============== BENCHMARK ==============

def run_rag_benchmark(model, tokenizer, test_data: List[Dict],
                      corpus: List[Dict], vectorizer, tfidf_matrix,
                      top_k_values: List[int], max_new_tokens: int = 512) -> Dict:
    """Run RAG ablation: test with different top-k values."""
    from benchmark import (
        calculate_citation_accuracy, calculate_hallucination_rate,
        calculate_key_info, calculate_length_ratio,
        detect_language, classify_domain,
    )

    configs = [{'name': 'FT only', 'top_k': 0}]
    for k in top_k_values:
        configs.append({'name': f'FT + RAG (top-{k})', 'top_k': k})

    all_results = {}

    for config in configs:
        print(f"\n--- {config['name']} ---")
        results = []
        total_time = 0

        for i, sample in enumerate(test_data):
            instruction = sample['instruction']
            input_text = sample.get('input', '')
            reference = sample['output']
            query = f"{instruction} {input_text}"

            # Get context
            if config['top_k'] > 0 and vectorizer is not None:
                docs = retrieve(query, vectorizer, tfidf_matrix, corpus, config['top_k'])
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

            results.append({
                'id': i,
                'language': detect_language(f"{instruction} {input_text} {reference}"),
                'domain': classify_domain(f"{instruction} {input_text} {reference}"),
                'metrics': metrics,
            })

            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(test_data)}")

        # Average metrics
        n = len(results)
        avg = {}
        for r in results:
            for k, v in r['metrics'].items():
                avg[k] = avg.get(k, 0) + v / n

        all_results[config['name']] = {
            'config': config,
            'num_samples': n,
            'total_time': total_time,
            'avg_metrics': avg,
            'results': results,
        }

        print(f"  CitAcc={avg['citation_accuracy']*100:.1f}%, "
              f"Halluc={avg['hallucination_rate']*100:.1f}%, "
              f"KeyInfo={avg['key_info']*100:.1f}%")

    return all_results


def print_rag_table(all_results: Dict):
    """Print RAG ablation comparison table."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 3: RAG ABLATION")
    print("=" * 80)

    metrics = ['citation_accuracy', 'hallucination_rate', 'key_info', 'length_ratio']
    short = {'citation_accuracy': 'Cit.Acc', 'hallucination_rate': 'Halluc.',
             'key_info': 'KeyInfo', 'length_ratio': 'LenRatio'}

    header = f"{'Configuration':<25} {'N':>4}"
    for m in metrics:
        header += f" {short[m]:>8}"
    header += f" {'Lat(s)':>7}"
    print(header)
    print("-" * len(header))

    for name, data in all_results.items():
        avg = data['avg_metrics']
        row = f"{name:<25} {data['num_samples']:>4}"
        for m in metrics:
            row += f" {avg.get(m, 0)*100:>7.1f}%"
        row += f" {avg.get('gen_time', 0):>7.2f}"
        print(row)


def main():
    parser = argparse.ArgumentParser(description='RAG Ablation Benchmark (Experiment 3)')
    parser.add_argument('--model', type=str, required=True, help='Fine-tuned model path')
    parser.add_argument('--corpus', type=str, required=True, help='Legal document corpus path')
    parser.add_argument('--samples', type=int, default=100, help='Number of test samples')
    parser.add_argument('--top-k', nargs='+', type=int, default=[1, 3, 5], help='Top-k values to test')
    parser.add_argument('--output', type=str, default=None, help='Output JSON path')
    parser.add_argument('--max-tokens', type=int, default=512, help='Max new tokens')
    args = parser.parse_args()

    from unsloth import FastLanguageModel

    print("=" * 60)
    print(f"RAG ABLATION BENCHMARK")
    print(f"Model: {args.model}")
    print(f"Corpus: {args.corpus}")
    print(f"Top-k: {args.top_k}")
    print("=" * 60)

    if not Path(args.model).exists():
        print(f"ERROR: Model not found: {args.model}")
        return

    # Load corpus
    corpus = load_corpus(args.corpus)
    if not corpus:
        print("ERROR: Empty corpus. Create a JSONL file with legal documents.")
        print('Format: {"id": "1", "title": "ТК РК ст. 28", "text": "..."}')
        return
    print(f"Corpus: {len(corpus)} documents")

    vectorizer, tfidf_matrix = build_index(corpus)

    # Load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model, max_seq_length=2048, dtype=None, load_in_4bit=True,
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

    # Run
    all_results = run_rag_benchmark(
        model, tokenizer, data, corpus, vectorizer, tfidf_matrix,
        args.top_k, args.max_tokens
    )

    print_rag_table(all_results)

    # Save
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"results/rag_ablation_{timestamp}.json"
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump({
            'model': args.model,
            'corpus_size': len(corpus),
            'top_k_values': args.top_k,
            'configs': {name: {k: v for k, v in data.items() if k != 'results'}
                        for name, data in all_results.items()},
        }, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved: {args.output}")


if __name__ == "__main__":
    main()
