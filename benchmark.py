"""
Подробный бенчмарк для оценки качества обученной модели Qwen3-14B
Сравнение с эталонными ответами из датасета
"""
import json
import time
import torch
import random
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
from collections import defaultdict

from unsloth import FastLanguageModel
from datasets import load_dataset
from tqdm import tqdm

# Метрики
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("⚠️  rouge_score не установлен. Установи: pip install rouge-score")

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except:
        nltk.download('punkt', quiet=True)
    BLEU_AVAILABLE = True
except ImportError:
    BLEU_AVAILABLE = False
    print("⚠️  nltk не установлен. Установи: pip install nltk")

# ============== КОНФИГУРАЦИЯ ==============

# Путь к обученной модели
MODEL_PATH = "finetuned_qwen3_8b"  # Обученная 8B модель

# Датасет для тестирования
VAL_FILE = "combined_data/validation.jsonl"

# Количество примеров для теста (None = все)
NUM_SAMPLES = 100  # Уменьшено для скорости, поставь None для полного теста

# Параметры генерации
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.1  # Низкая температура для детерминированных ответов
TOP_P = 0.9

# Промпт шаблон (должен совпадать с обучением)
ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

# ============== МЕТРИКИ ==============

def calculate_exact_match(pred: str, reference: str) -> float:
    """Точное совпадение (нормализованное)"""
    pred_norm = pred.strip().lower()
    ref_norm = reference.strip().lower()
    return 1.0 if pred_norm == ref_norm else 0.0

def calculate_token_overlap(pred: str, reference: str) -> float:
    """F1 score по токенам (слова)"""
    pred_tokens = set(pred.lower().split())
    ref_tokens = set(reference.lower().split())
    
    if not pred_tokens or not ref_tokens:
        return 0.0
    
    common = pred_tokens & ref_tokens
    
    if len(common) == 0:
        return 0.0
    
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1

def calculate_rouge(pred: str, reference: str, scorer) -> Dict[str, float]:
    """ROUGE scores"""
    scores = scorer.score(reference, pred)
    return {
        'rouge1': scores['rouge1'].fmeasure,
        'rouge2': scores['rouge2'].fmeasure,
        'rougeL': scores['rougeL'].fmeasure,
    }

def calculate_bleu(pred: str, reference: str) -> float:
    """BLEU score"""
    pred_tokens = pred.lower().split()
    ref_tokens = [reference.lower().split()]
    
    if not pred_tokens or not ref_tokens[0]:
        return 0.0
    
    smoothing = SmoothingFunction().method1
    try:
        score = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smoothing)
    except:
        score = 0.0
    return score

def calculate_length_ratio(pred: str, reference: str) -> float:
    """Соотношение длин (ближе к 1 = лучше)"""
    pred_len = len(pred.split())
    ref_len = len(reference.split())
    
    if ref_len == 0:
        return 0.0
    
    ratio = pred_len / ref_len
    # Штраф за слишком короткие или длинные ответы
    if ratio > 1:
        return 1 / ratio
    return ratio

def calculate_contains_key_info(pred: str, reference: str) -> float:
    """Проверка наличия ключевой информации"""
    # Извлекаем числа, даты, ключевые термины из reference
    import re
    
    # Числа
    ref_numbers = set(re.findall(r'\d+', reference))
    pred_numbers = set(re.findall(r'\d+', pred))
    
    if ref_numbers:
        number_overlap = len(ref_numbers & pred_numbers) / len(ref_numbers)
    else:
        number_overlap = 1.0
    
    # Ключевые юридические термины (на русском)
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

# ============== ОСНОВНЫЕ ФУНКЦИИ ==============

def load_model(model_path: str):
    """Загрузка модели"""
    print(f"🔄 Загрузка модели: {model_path}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,  # 4-bit для инференса (быстрее)
    )
    
    FastLanguageModel.for_inference(model)
    print("✅ Модель загружена")
    
    return model, tokenizer

def load_test_data(val_file: str, num_samples: int = None) -> List[Dict]:
    """Загрузка тестовых данных"""
    print(f"📊 Загрузка тестовых данных: {val_file}")
    
    data = []
    with open(val_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    print(f"   Всего записей: {len(data)}")
    
    if num_samples and num_samples < len(data):
        random.seed(42)  # Для воспроизводимости
        data = random.sample(data, num_samples)
        print(f"   Выбрано для теста: {num_samples}")
    
    return data

def generate_response(model, tokenizer, instruction: str, input_text: str) -> Tuple[str, float]:
    """Генерация ответа"""
    prompt = ALPACA_PROMPT.format(instruction, input_text, "")
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    gen_time = time.time() - start_time
    
    # Декодирование только сгенерированной части
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    return response.strip(), gen_time

def run_benchmark(model, tokenizer, test_data: List[Dict]) -> Dict:
    """Запуск бенчмарка"""
    print("\n" + "=" * 60)
    print("🚀 ЗАПУСК БЕНЧМАРКА")
    print("=" * 60)
    
    # Инициализация скореров
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False) if ROUGE_AVAILABLE else None
    
    results = []
    metrics_sum = defaultdict(float)
    
    total_gen_time = 0
    
    for i, sample in enumerate(tqdm(test_data, desc="Тестирование")):
        instruction = sample['instruction']
        input_text = sample.get('input', '')
        reference = sample['output']
        
        # Генерация
        prediction, gen_time = generate_response(model, tokenizer, instruction, input_text)
        total_gen_time += gen_time
        
        # Расчёт метрик
        metrics = {
            'exact_match': calculate_exact_match(prediction, reference),
            'token_f1': calculate_token_overlap(prediction, reference),
            'length_ratio': calculate_length_ratio(prediction, reference),
            'key_info': calculate_contains_key_info(prediction, reference),
        }
        
        if ROUGE_AVAILABLE:
            rouge_scores = calculate_rouge(prediction, reference, rouge)
            metrics.update(rouge_scores)
        
        if BLEU_AVAILABLE:
            metrics['bleu'] = calculate_bleu(prediction, reference)
        
        metrics['gen_time'] = gen_time
        
        # Суммирование для средних
        for k, v in metrics.items():
            metrics_sum[k] += v
        
        # Сохранение результата
        results.append({
            'id': i,
            'instruction': instruction[:100] + '...' if len(instruction) > 100 else instruction,
            'input': input_text[:50] + '...' if len(input_text) > 50 else input_text,
            'reference': reference[:200] + '...' if len(reference) > 200 else reference,
            'prediction': prediction[:200] + '...' if len(prediction) > 200 else prediction,
            'metrics': metrics,
        })
    
    # Расчёт средних
    num_samples = len(test_data)
    avg_metrics = {k: v / num_samples for k, v in metrics_sum.items()}
    
    return {
        'num_samples': num_samples,
        'total_time': total_gen_time,
        'avg_time_per_sample': total_gen_time / num_samples,
        'avg_metrics': avg_metrics,
        'results': results,
    }

def print_results(benchmark_results: Dict):
    """Вывод результатов"""
    print("\n" + "=" * 60)
    print("📊 РЕЗУЛЬТАТЫ БЕНЧМАРКА")
    print("=" * 60)
    
    num_samples = benchmark_results['num_samples']
    total_time = benchmark_results['total_time']
    avg_metrics = benchmark_results['avg_metrics']
    
    print(f"\n📈 ОБЩАЯ СТАТИСТИКА:")
    print(f"   Протестировано примеров: {num_samples}")
    print(f"   Общее время генерации: {total_time:.1f} сек")
    print(f"   Среднее время на пример: {total_time/num_samples:.2f} сек")
    print(f"   Примеров в минуту: {num_samples / (total_time / 60):.1f}")
    
    print(f"\n📊 МЕТРИКИ КАЧЕСТВА:")
    print("-" * 40)
    
    # Основные метрики
    print(f"   Token F1:        {avg_metrics['token_f1']*100:.2f}%")
    print(f"   Exact Match:     {avg_metrics['exact_match']*100:.2f}%")
    print(f"   Key Info Score:  {avg_metrics['key_info']*100:.2f}%")
    print(f"   Length Ratio:    {avg_metrics['length_ratio']*100:.2f}%")
    
    if ROUGE_AVAILABLE:
        print("-" * 40)
        print(f"   ROUGE-1:         {avg_metrics['rouge1']*100:.2f}%")
        print(f"   ROUGE-2:         {avg_metrics['rouge2']*100:.2f}%")
        print(f"   ROUGE-L:         {avg_metrics['rougeL']*100:.2f}%")
    
    if BLEU_AVAILABLE:
        print("-" * 40)
        print(f"   BLEU:            {avg_metrics['bleu']*100:.2f}%")
    
    # Интерпретация результатов
    print("\n📝 ИНТЕРПРЕТАЦИЯ:")
    print("-" * 40)
    
    overall_score = (
        avg_metrics['token_f1'] * 0.3 +
        avg_metrics['key_info'] * 0.3 +
        avg_metrics.get('rougeL', avg_metrics['token_f1']) * 0.2 +
        avg_metrics['length_ratio'] * 0.2
    )
    
    print(f"   Общий балл: {overall_score*100:.1f}%")
    
    if overall_score >= 0.7:
        print("   ✅ Отличное качество! Модель хорошо обучена.")
    elif overall_score >= 0.5:
        print("   ⚠️  Хорошее качество. Возможны улучшения.")
    elif overall_score >= 0.3:
        print("   ⚠️  Среднее качество. Рекомендуется дообучение.")
    else:
        print("   ❌ Низкое качество. Требуется переобучение.")
    
    print("=" * 60)

def save_results(benchmark_results: Dict, output_file: str = None):
    """Сохранение результатов в файл"""
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"benchmark_results_{timestamp}.json"
    
    # Подготовка данных для сохранения
    save_data = {
        'model_path': MODEL_PATH,
        'num_samples': benchmark_results['num_samples'],
        'total_time': benchmark_results['total_time'],
        'avg_metrics': benchmark_results['avg_metrics'],
        'timestamp': datetime.now().isoformat(),
        'config': {
            'max_new_tokens': MAX_NEW_TOKENS,
            'temperature': TEMPERATURE,
            'top_p': TOP_P,
        },
        # Детальные результаты
        'detailed_results': benchmark_results['results'][:20],  # Первые 20 для экономии места
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Результаты сохранены: {output_file}")

def show_examples(benchmark_results: Dict, num_examples: int = 5):
    """Показать примеры предсказаний"""
    print("\n" + "=" * 60)
    print("📝 ПРИМЕРЫ ПРЕДСКАЗАНИЙ")
    print("=" * 60)
    
    results = benchmark_results['results']
    
    # Случайные примеры
    examples = random.sample(results, min(num_examples, len(results)))
    
    for i, ex in enumerate(examples, 1):
        print(f"\n--- Пример {i} ---")
        print(f"📝 Instruction: {ex['instruction']}")
        if ex['input']:
            print(f"📥 Input: {ex['input']}")
        print(f"\n✅ Эталон:")
        print(f"   {ex['reference']}")
        print(f"\n🤖 Предсказание:")
        print(f"   {ex['prediction']}")
        print(f"\n📊 Метрики:")
        print(f"   Token F1: {ex['metrics']['token_f1']*100:.1f}%")
        print(f"   Key Info: {ex['metrics']['key_info']*100:.1f}%")
        if 'rougeL' in ex['metrics']:
            print(f"   ROUGE-L: {ex['metrics']['rougeL']*100:.1f}%")
        print("-" * 40)

def show_worst_examples(benchmark_results: Dict, num_examples: int = 5):
    """Показать худшие примеры (для анализа ошибок)"""
    print("\n" + "=" * 60)
    print("❌ ХУДШИЕ ПРЕДСКАЗАНИЯ (для анализа)")
    print("=" * 60)
    
    results = benchmark_results['results']
    
    # Сортировка по token_f1
    sorted_results = sorted(results, key=lambda x: x['metrics']['token_f1'])
    worst = sorted_results[:num_examples]
    
    for i, ex in enumerate(worst, 1):
        print(f"\n--- Худший пример {i} ---")
        print(f"📝 Instruction: {ex['instruction']}")
        if ex['input']:
            print(f"📥 Input: {ex['input']}")
        print(f"\n✅ Эталон:")
        print(f"   {ex['reference']}")
        print(f"\n🤖 Предсказание:")
        print(f"   {ex['prediction']}")
        print(f"\n📊 Token F1: {ex['metrics']['token_f1']*100:.1f}%")
        print("-" * 40)

def main():
    print("=" * 60)
    print("🧪 БЕНЧМАРК МОДЕЛИ QWEN3-14B")
    print("=" * 60)
    print(f"Модель: {MODEL_PATH}")
    print(f"Датасет: {VAL_FILE}")
    print(f"Примеров: {NUM_SAMPLES or 'все'}")
    print("=" * 60)
    
    # Проверка наличия модели
    if not Path(MODEL_PATH).exists():
        print(f"❌ Модель не найдена: {MODEL_PATH}")
        print("   Укажи правильный путь в MODEL_PATH")
        return
    
    # Загрузка
    model, tokenizer = load_model(MODEL_PATH)
    test_data = load_test_data(VAL_FILE, NUM_SAMPLES)
    
    # Бенчмарк
    results = run_benchmark(model, tokenizer, test_data)
    
    # Результаты
    print_results(results)
    show_examples(results, num_examples=3)
    show_worst_examples(results, num_examples=3)
    
    # Сохранение
    save_results(results)
    
    print("\n✅ Бенчмарк завершён!")

if __name__ == "__main__":
    main()

