"""
Prepare blind evaluation sheets for human (lawyer) evaluation.
Experiment 5: Human Evaluation.

Selects representative questions, generates responses from multiple models,
and creates a CSV/JSON evaluation sheet where lawyers rate quality blindly.

Usage:
    python prepare_human_eval.py --models lora_qwen3_4b lora_qwen3_8b --baselines unsloth/Qwen3-8B-unsloth-bnb-4bit
    python prepare_human_eval.py --from-results results/benchmark_4b.json results/baseline_8b.json
"""
import json
import csv
import random
import argparse
from pathlib import Path
from datetime import datetime

from config import VAL_FILE


def load_val_data(num_samples: int = 50) -> list:
    """Load and sample validation data."""
    data = []
    with open(VAL_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    random.seed(42)
    return random.sample(data, min(num_samples, len(data)))


def prepare_from_results(result_files: list, num_questions: int = 50):
    """Build evaluation sheet from existing benchmark result JSONs.
    Each question gets answers from all models, shuffled and anonymized."""

    all_results = []
    for path in result_files:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            tag = "Base" if data.get('is_baseline') else "FT"
            model = data['model']
            # Extract size
            size = "?"
            for s in ['4b', '8b', '14b', '32b']:
                if s in model.lower():
                    size = s.upper()
            label = f"Qwen3-{size} {tag}"
            all_results.append({
                'label': label,
                'model': model,
                'is_baseline': data.get('is_baseline', False),
                'results': {r['id']: r for r in data.get('detailed_results', [])},
            })

    if not all_results:
        print("No results loaded!")
        return

    random.seed(42)

    # Get question IDs present in ALL result files
    common_ids = set(all_results[0]['results'].keys())
    for r in all_results[1:]:
        common_ids &= set(r['results'].keys())
    common_ids = sorted(common_ids)[:num_questions]

    print(f"Models: {len(all_results)}, Common questions: {len(common_ids)}")

    # Build evaluation rows
    eval_rows = []
    # Mapping from anonymous ID to real model (kept secret from evaluators)
    model_key = {}

    for qid in common_ids:
        # Get reference from first result
        ref_data = all_results[0]['results'][qid]
        question = ref_data.get('instruction', '')
        context = ref_data.get('input', '')
        reference = ref_data.get('reference', '')

        # Collect all model answers
        answers = []
        for model_data in all_results:
            r = model_data['results'].get(qid, {})
            answers.append({
                'model_label': model_data['label'],
                'prediction': r.get('prediction', '(no answer)'),
            })

        # Shuffle answers for blind evaluation
        random.shuffle(answers)

        for idx, ans in enumerate(answers):
            anon_id = chr(65 + idx)  # A, B, C, D, ...
            model_key[f"Q{qid}_{anon_id}"] = ans['model_label']

            eval_rows.append({
                'question_id': qid,
                'question': question,
                'context': context,
                'reference_answer': reference,
                'answer_id': anon_id,
                'model_answer': ans['prediction'],
                # Evaluator fills these:
                'correctness_1_5': '',
                'completeness_1_5': '',
                'relevance_1_5': '',
                'hallucination_yes_no': '',
                'notes': '',
            })

    # Save evaluation sheet
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("human_eval")
    out_dir.mkdir(exist_ok=True)

    # CSV with Russian headers for evaluators
    ru_fieldnames = [
        'номер_вопроса', 'вопрос', 'контекст', 'эталонный_ответ',
        'вариант', 'ответ_модели',
        'корректность_1_5', 'полнота_1_5', 'релевантность_1_5',
        'галлюцинация_да_нет', 'комментарий'
    ]
    csv_file = out_dir / f"eval_sheet_{timestamp}.csv"
    with open(csv_file, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(ru_fieldnames)
        for row in eval_rows:
            writer.writerow([
                row['question_id'], row['question'], row['context'],
                row['reference_answer'], row['answer_id'], row['model_answer'],
                '', '', '', '', ''
            ])

    # English CSV (for processing)
    csv_en_file = out_dir / f"eval_sheet_en_{timestamp}.csv"
    with open(csv_en_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=eval_rows[0].keys())
        writer.writeheader()
        writer.writerows(eval_rows)

    # Secret key (model mapping)
    key_file = out_dir / f"model_key_{timestamp}.json"
    with open(key_file, 'w', encoding='utf-8') as f:
        json.dump(model_key, f, ensure_ascii=False, indent=2)

    # JSON version
    json_file = out_dir / f"eval_sheet_{timestamp}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(eval_rows, f, ensure_ascii=False, indent=2)

    # Instructions file for evaluators (Russian)
    instr_file = out_dir / "ИНСТРУКЦИЯ.txt"
    with open(instr_file, 'w', encoding='utf-8') as f:
        f.write("""ИНСТРУКЦИЯ ДЛЯ ЭКСПЕРТНОЙ ОЦЕНКИ ОТВЕТОВ ИИ-МОДЕЛИ
===================================================

Вам предоставлены 50 юридических вопросов по казахстанскому законодательству.
На каждый вопрос даны 4 ответа (A, B, C, D) от РАЗНЫХ моделей.
Вы НЕ знаете, какая модель сгенерировала какой ответ.

Для каждого ответа оцените по шкале от 1 до 5:

1. КОРРЕКТНОСТЬ (корректность_1_5):
   1 = полностью неверный ответ, ошибки в законодательстве
   2 = частично верный, но с существенными ошибками
   3 = в целом верный, но есть неточности
   4 = верный ответ с незначительными неточностями
   5 = полностью корректный и точный ответ

2. ПОЛНОТА (полнота_1_5):
   1 = ответ не раскрывает вопрос
   2 = ответ поверхностный, упущены ключевые аспекты
   3 = основные аспекты раскрыты, но есть пробелы
   4 = ответ достаточно полный
   5 = исчерпывающий ответ, все аспекты раскрыты

3. РЕЛЕВАНТНОСТЬ (релевантность_1_5):
   1 = ответ не по теме
   2 = частично по теме
   3 = по теме, но есть лишняя информация
   4 = релевантный ответ
   5 = точно по теме, без лишнего

4. ГАЛЛЮЦИНАЦИЯ (галлюцинация_да_нет):
   да = ответ содержит выдуманные нормы, несуществующие статьи или ложные ссылки
   нет = все ссылки и нормы корректны (или ответ не содержит ссылок)

5. КОММЕНТАРИЙ (необязательно):
   Любые замечания к ответу.

ВАЖНО:
- Сравнивайте ответ модели с ЭТАЛОННЫМ ОТВЕТОМ
- Оценивайте объективно, не зная какая модель дала ответ
- Если ответ на другом языке (не RU/KZ) — ставьте 1 по всем критериям
""")

    print(f"\nEvaluation sheet (RU): {csv_file}")
    print(f"Evaluation sheet (EN): {csv_en_file}")
    print(f"Model key (secret):    {key_file}")
    print(f"Instructions:          {instr_file}")
    print(f"\nTotal rows: {len(eval_rows)} ({len(common_ids)} questions x {len(all_results)} models)")
    print(f"\nДля юристов: отправьте {csv_file} + ИНСТРУКЦИЯ.txt")
    print(f"model_key — НЕ отправлять юристам (секретный ключ)")


def prepare_live(model_paths: list, baseline_paths: list, num_questions: int = 50):
    """Generate responses live from models (requires GPU). Full text, no truncation."""
    from unsloth import FastLanguageModel
    from config import ALPACA_PROMPT
    from tqdm import tqdm

    val_data = load_val_data(num_questions)
    all_models = []

    for path in (model_paths or []):
        all_models.append(('FT', path))
    for path in (baseline_paths or []):
        all_models.append(('Base', path))

    if not all_models:
        print("No models specified!")
        return

    # Generate responses from each model
    result_files = []
    for tag, model_path in all_models:
        print(f"\nLoading {tag}: {model_path}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path, max_seq_length=2048, dtype=None, load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)

        import torch
        results = {}
        for i, sample in enumerate(tqdm(val_data, desc=f"Generating ({tag}: {Path(model_path).name})")):
            prompt = ALPACA_PROMPT.format(sample['instruction'], sample.get('input', ''), "")
            inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=1024,
                                         temperature=0.1, top_p=0.9, use_cache=True,
                                         pad_token_id=tokenizer.pad_token_id)
            response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            results[i] = {
                'id': i,
                'instruction': sample['instruction'],
                'input': sample.get('input', ''),
                'reference': sample['output'],
                'prediction': response.strip(),
            }

        # Save as temporary result file
        tmp = {
            'model': model_path,
            'is_baseline': tag == 'Base',
            'detailed_results': list(results.values()),
        }
        tmp_file = f"/tmp/human_eval_{tag}_{Path(model_path).name}.json"
        with open(tmp_file, 'w') as f:
            json.dump(tmp, f, ensure_ascii=False, indent=2)
        result_files.append(tmp_file)

        # Free GPU memory
        del model, tokenizer
        torch.cuda.empty_cache()

    # Now build evaluation sheet from these results
    prepare_from_results(result_files, num_questions)


def main():
    parser = argparse.ArgumentParser(description='Prepare human evaluation sheets')
    parser.add_argument('--from-results', nargs='+', help='Existing benchmark result JSON files')
    parser.add_argument('--models', nargs='+', help='Fine-tuned model paths (live generation)')
    parser.add_argument('--baselines', nargs='+', help='Baseline model names (live generation)')
    parser.add_argument('--questions', type=int, default=50, help='Number of questions (default: 50)')
    args = parser.parse_args()

    if args.from_results:
        prepare_from_results(args.from_results, args.questions)
    elif args.models or args.baselines:
        prepare_live(args.models, args.baselines, args.questions)
    else:
        print("Specify --from-results OR --models/--baselines")
        print("\nExample:")
        print("  python prepare_human_eval.py --from-results results/*.json")
        print("  python prepare_human_eval.py --models lora_qwen3_8b --baselines unsloth/Qwen3-8B-unsloth-bnb-4bit")


if __name__ == "__main__":
    main()
