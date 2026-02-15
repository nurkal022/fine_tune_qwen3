"""
Benchmark for evaluating fine-tuned models on Kazakhstan legal domain.
Supports both fine-tuned and baseline (untuned) model evaluation.

Usage:
    python benchmark.py --model lora_qwen3_8b
    python benchmark.py --model lora_qwen3_4b --samples 200
    python benchmark.py --baseline unsloth/Qwen3-8B-unsloth-bnb-4bit
    python benchmark.py --model lora_qwen3_8b --output results/8b_finetuned.json
"""
import json
import re
import time
import argparse
import torch
import random
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Set
from collections import defaultdict

from unsloth import FastLanguageModel
from tqdm import tqdm

from config import ALPACA_PROMPT, VAL_FILE

# Optional metrics
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


# ============== LEGAL CITATION EXTRACTION ==============

def extract_legal_citations(text: str) -> Set[str]:
    """Extract legal citations from text (article numbers, law references)."""
    citations = set()
    text_lower = text.lower()

    # "статья 123", "ст. 45", "статьи 12, 13"
    for m in re.finditer(r'(?:стать[яиейю]|ст\.?)\s*(\d+)', text_lower):
        citations.add(f"ст.{m.group(1)}")

    # "пункт 3", "п. 2", "подпункт 1)"
    for m in re.finditer(r'(?:пункт[а-я]*|п\.?)\s*(\d+)', text_lower):
        citations.add(f"п.{m.group(1)}")

    # "Закон №123", "Закон от 12.03.2020"
    for m in re.finditer(r'закон[а-я]*\s*(?:№|номер)?\s*(\d[\d\-\.]*\d)', text_lower):
        citations.add(f"закон_{m.group(1)}")

    # "Кодекс" references with article
    for m in re.finditer(r'кодекс[а-я]*.*?стать[яиейю]\s*(\d+)', text_lower):
        citations.add(f"ст.{m.group(1)}")

    # Standalone article-like numbers "123-IV", "45-ІІ"
    for m in re.finditer(r'\b(\d+)[\-–](I{1,4}V?|V?I{0,4}|[IVX]+)\b', text):
        citations.add(f"ref_{m.group(0)}")

    return citations


# ============== METRICS ==============

def calculate_rouge(pred: str, reference: str, scorer) -> Dict[str, float]:
    """ROUGE scores."""
    scores = scorer.score(reference, pred)
    return {
        'rouge1': scores['rouge1'].fmeasure,
        'rouge2': scores['rouge2'].fmeasure,
        'rougeL': scores['rougeL'].fmeasure,
    }


def calculate_bleu(pred: str, reference: str) -> float:
    """BLEU score."""
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
    """Check if legal citations from reference appear in prediction."""
    ref_citations = extract_legal_citations(reference)
    if not ref_citations:
        return 1.0  # no citations to check
    pred_citations = extract_legal_citations(pred)
    matched = ref_citations & pred_citations
    return len(matched) / len(ref_citations)


def calculate_hallucination_rate(pred: str, reference: str) -> float:
    """Detect fabricated legal citations not present in reference."""
    pred_citations = extract_legal_citations(pred)
    if not pred_citations:
        return 0.0  # no citations = no hallucination
    ref_citations = extract_legal_citations(reference)
    hallucinated = pred_citations - ref_citations
    return len(hallucinated) / len(pred_citations)


def calculate_key_info(pred: str, reference: str) -> float:
    """Check overlap of key information (numbers + legal terms)."""
    # Numbers
    ref_numbers = set(re.findall(r'\d+', reference))
    pred_numbers = set(re.findall(r'\d+', pred))
    if ref_numbers:
        number_overlap = len(ref_numbers & pred_numbers) / len(ref_numbers)
    else:
        number_overlap = 1.0

    # Legal terms
    legal_terms = [
        'статья', 'закон', 'кодекс', 'право', 'обязанность', 'договор',
        'срок', 'штраф', 'суд', 'иск', 'заявление', 'документ', 'лицо',
        'гражданин', 'организация', 'государство', 'орган', 'решение'
    ]
    ref_lower = reference.lower()
    pred_lower = pred.lower()
    ref_terms = [t for t in legal_terms if t in ref_lower]
    if ref_terms:
        term_overlap = sum(1 for t in ref_terms if t in pred_lower) / len(ref_terms)
    else:
        term_overlap = 1.0

    return (number_overlap + term_overlap) / 2


def calculate_length_ratio(pred: str, reference: str) -> float:
    """Length ratio (closer to 1 = better)."""
    pred_len = len(pred.split())
    ref_len = len(reference.split())
    if ref_len == 0:
        return 0.0
    ratio = pred_len / ref_len
    return 1 / ratio if ratio > 1 else ratio


# ============== MODEL LOADING & INFERENCE ==============

def load_model(model_path: str, is_baseline: bool = False):
    """Load model for evaluation."""
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
                      max_new_tokens: int = 512) -> Tuple[str, float]:
    """Generate a response and measure time."""
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
    """Compute BERTScore for a batch (more efficient than per-sample)."""
    P, R, F1 = bert_score_fn(predictions, references, lang="ru", verbose=False)
    return F1.tolist()


def run_benchmark(model, tokenizer, test_data: List[Dict],
                  max_new_tokens: int = 512) -> Dict:
    """Run full benchmark."""
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

        prediction, gen_time = generate_response(model, tokenizer, instruction, input_text, max_new_tokens)
        total_gen_time += gen_time

        # Per-sample metrics
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

        # Collect for batch BERTScore
        if BERTSCORE_AVAILABLE:
            predictions_for_bert.append(prediction)
            references_for_bert.append(reference)

        for k, v in metrics.items():
            metrics_sum[k] += v

        results.append({
            'id': i,
            'instruction': instruction[:100] + '...' if len(instruction) > 100 else instruction,
            'input': input_text[:50] + '...' if len(input_text) > 50 else input_text,
            'reference': reference[:200] + '...' if len(reference) > 200 else reference,
            'prediction': prediction[:200] + '...' if len(prediction) > 200 else prediction,
            'metrics': metrics,
        })

    # Batch BERTScore
    if BERTSCORE_AVAILABLE and predictions_for_bert:
        print("Computing BERTScore...")
        bert_scores = compute_bertscore_batch(predictions_for_bert, references_for_bert)
        for i, score in enumerate(bert_scores):
            results[i]['metrics']['bertscore_f1'] = score
            metrics_sum['bertscore_f1'] += score

    num_samples = len(test_data)
    avg_metrics = {k: v / num_samples for k, v in metrics_sum.items()}

    return {
        'num_samples': num_samples,
        'total_time': total_gen_time,
        'avg_time_per_sample': total_gen_time / num_samples,
        'avg_metrics': avg_metrics,
        'results': results,
    }


def print_results(benchmark_results: Dict, model_name: str, is_baseline: bool):
    """Print benchmark results."""
    print("\n" + "=" * 60)
    print(f"BENCHMARK RESULTS {'(BASELINE)' if is_baseline else '(FINE-TUNED)'}")
    print(f"Model: {model_name}")
    print("=" * 60)

    num_samples = benchmark_results['num_samples']
    total_time = benchmark_results['total_time']
    avg = benchmark_results['avg_metrics']

    print(f"\nSTATISTICS:")
    print(f"  Samples: {num_samples}")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Avg time/sample: {total_time/num_samples:.2f}s")
    print(f"  Throughput: {num_samples / (total_time / 60):.1f} samples/min")

    print(f"\nMETRICS:")
    print("-" * 40)

    if BERTSCORE_AVAILABLE and 'bertscore_f1' in avg:
        print(f"  BERTScore F1:       {avg['bertscore_f1']*100:.2f}%")

    if ROUGE_AVAILABLE and 'rougeL' in avg:
        print(f"  ROUGE-1:            {avg['rouge1']*100:.2f}%")
        print(f"  ROUGE-2:            {avg['rouge2']*100:.2f}%")
        print(f"  ROUGE-L:            {avg['rougeL']*100:.2f}%")

    if BLEU_AVAILABLE and 'bleu' in avg:
        print(f"  BLEU:               {avg['bleu']*100:.2f}%")

    print("-" * 40)
    print(f"  Citation Accuracy:  {avg['citation_accuracy']*100:.2f}%")
    print(f"  Hallucination Rate: {avg['hallucination_rate']*100:.2f}%")
    print(f"  Key Info Score:     {avg['key_info']*100:.2f}%")
    print(f"  Length Ratio:       {avg['length_ratio']*100:.2f}%")
    print("=" * 60)


def save_results(benchmark_results: Dict, model_name: str,
                 is_baseline: bool, output_file: str = None):
    """Save results to JSON."""
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = model_name.replace("/", "_")
        prefix = "baseline" if is_baseline else "benchmark"
        output_file = f"{prefix}_{safe_name}_{timestamp}.json"

    # Ensure output directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_data = {
        'model': model_name,
        'is_baseline': is_baseline,
        'num_samples': benchmark_results['num_samples'],
        'total_time': benchmark_results['total_time'],
        'avg_metrics': benchmark_results['avg_metrics'],
        'timestamp': datetime.now().isoformat(),
        'detailed_results': benchmark_results['results'][:20],
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved: {output_file}")


def show_examples(benchmark_results: Dict, num_examples: int = 3):
    """Show sample predictions."""
    print("\n" + "=" * 60)
    print("SAMPLE PREDICTIONS")
    print("=" * 60)

    results = benchmark_results['results']
    examples = random.sample(results, min(num_examples, len(results)))

    for i, ex in enumerate(examples, 1):
        print(f"\n--- Example {i} ---")
        print(f"Instruction: {ex['instruction']}")
        if ex['input']:
            print(f"Input: {ex['input']}")
        print(f"\nReference:")
        print(f"  {ex['reference']}")
        print(f"\nPrediction:")
        print(f"  {ex['prediction']}")
        print(f"\nMetrics:")
        if 'bertscore_f1' in ex['metrics']:
            print(f"  BERTScore F1: {ex['metrics']['bertscore_f1']*100:.1f}%")
        print(f"  Citation Acc: {ex['metrics']['citation_accuracy']*100:.1f}%")
        print(f"  Key Info: {ex['metrics']['key_info']*100:.1f}%")
        if 'rougeL' in ex['metrics']:
            print(f"  ROUGE-L: {ex['metrics']['rougeL']*100:.1f}%")
        print("-" * 40)


# ============== MAIN ==============

def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark Qwen3 models on legal QA')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--model', type=str, help='Path to fine-tuned LoRA model')
    group.add_argument('--baseline', type=str, help='Base model name for baseline evaluation')
    parser.add_argument('--samples', type=int, default=100, help='Number of test samples (default: 100)')
    parser.add_argument('--output', type=str, default=None, help='Output JSON path')
    parser.add_argument('--max-tokens', type=int, default=512, help='Max new tokens (default: 512)')
    return parser.parse_args()


def main():
    args = parse_args()

    is_baseline = args.baseline is not None
    model_path = args.baseline if is_baseline else args.model

    print("=" * 60)
    print(f"BENCHMARK {'(BASELINE)' if is_baseline else '(FINE-TUNED)'}")
    print(f"Model: {model_path}")
    print(f"Samples: {args.samples}")
    print("=" * 60)

    # Check model exists (for local paths)
    if not is_baseline and not Path(model_path).exists():
        print(f"ERROR: Model not found: {model_path}")
        return

    # Load
    model, tokenizer = load_model(model_path, is_baseline)

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
        print(f"  Selected for test: {args.samples}")

    # Run
    results = run_benchmark(model, tokenizer, data, args.max_tokens)

    # Output
    print_results(results, model_path, is_baseline)
    show_examples(results)
    save_results(results, model_path, is_baseline, args.output)

    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()
