"""
Извлечение юридических терминов для глоссария
"""
import json
import re
from collections import Counter
from pathlib import Path

# Казахские специфичные буквы
KK_CHARS = set("әғқңөұүһіӘҒҚҢӨҰҮҺІ")

def load_jsonl(path):
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records

def extract_text(records):
    """Собираем весь текст"""
    texts = []
    for r in records:
        texts.append(r.get("instruction", ""))
        texts.append(r.get("input", ""))
        texts.append(r.get("output", ""))
    return " ".join(texts)

def has_kk_chars(word):
    return bool(set(word) & KK_CHARS)

def is_abbreviation(word):
    """Аббревиатуры: ТОО, РК, ЖШС, КоАП и т.д."""
    if len(word) < 2 or len(word) > 10:
        return False
    # Полностью заглавные
    if word.isupper() and len(word) >= 2:
        return True
    # Смешанные типа КоАП, ГПХ
    upper_count = sum(1 for c in word if c.isupper())
    if upper_count >= 2 and len(word) <= 6:
        return True
    return False

def extract_terms(text):
    """Извлекаем слова и биграммы"""
    # Токенизация - слова кириллицы
    words = re.findall(r'[А-Яа-яӘәҒғҚқҢңӨөҰұҮүҺһІі]+', text)
    
    # Счётчик слов
    word_freq = Counter(words)
    
    # Биграммы (термины из 2 слов)
    bigrams = []
    for i in range(len(words) - 1):
        bigrams.append(f"{words[i]} {words[i+1]}")
    bigram_freq = Counter(bigrams)
    
    return word_freq, bigram_freq

def filter_candidates(word_freq, bigram_freq, min_freq=3):
    """Фильтруем кандидатов в термины"""
    
    candidates = {
        "abbreviations": [],      # Аббревиатуры
        "kazakh_specific": [],    # Казахские слова
        "legal_unigrams": [],     # Возможные юр. термины (1 слово)
        "legal_bigrams": [],      # Возможные юр. термины (2 слова)
    }
    
    # Стоп-слова (частые, но не термины)
    stopwords = {
        'в', 'на', 'по', 'с', 'и', 'или', 'а', 'не', 'для', 'от', 'к', 'из',
        'что', 'как', 'это', 'при', 'за', 'до', 'об', 'также', 'если', 'то',
        'быть', 'были', 'был', 'была', 'может', 'могут', 'должен', 'должны',
        'его', 'её', 'их', 'ее', 'он', 'она', 'они', 'мы', 'вы', 'этот', 'эта',
        # Казахские стоп-слова
        'бір', 'және', 'бойынша', 'үшін', 'мен', 'бұл', 'осы', 'сол',
        'болып', 'болады', 'бар', 'жоқ', 'де', 'да', 'немесе',
    }
    
    # Юридические паттерны
    legal_patterns = [
        r'кодекс', r'закон', r'статья', r'пункт', r'договор', r'право',
        r'орган', r'суд', r'акт', r'норм', r'регул', r'ответствен',
        r'обязат', r'полномоч', r'компетенц', r'бюджет', r'налог',
        # Казахские
        r'заң', r'құқық', r'орган', r'сот', r'кодекс', r'бап',
    ]
    legal_regex = re.compile('|'.join(legal_patterns), re.IGNORECASE)
    
    # 1. Аббревиатуры
    for word, freq in word_freq.items():
        if freq >= min_freq and is_abbreviation(word):
            candidates["abbreviations"].append((word, freq))
    
    # 2. Казахские слова (с специфичными буквами)
    for word, freq in word_freq.items():
        if freq >= min_freq and has_kk_chars(word) and len(word) >= 3:
            if word.lower() not in stopwords:
                candidates["kazakh_specific"].append((word, freq))
    
    # 3. Юридические термины (униграммы)
    for word, freq in word_freq.items():
        if freq >= min_freq and len(word) >= 5:
            if word.lower() not in stopwords and not is_abbreviation(word):
                if legal_regex.search(word):
                    candidates["legal_unigrams"].append((word, freq))
    
    # 4. Биграммы
    for bigram, freq in bigram_freq.items():
        if freq >= min_freq:
            words = bigram.split()
            # Хотя бы одно слово длинное
            if any(len(w) >= 4 for w in words):
                # Не стоп-слова оба
                if not all(w.lower() in stopwords for w in words):
                    candidates["legal_bigrams"].append((bigram, freq))
    
    # Сортируем по частоте
    for key in candidates:
        candidates[key] = sorted(candidates[key], key=lambda x: -x[1])
    
    return candidates

def main():
    # Загрузка
    train = load_jsonl("combined_data/train.jsonl")
    val = load_jsonl("combined_data/validation.jsonl")
    
    print(f"Загружено: {len(train)} train, {len(val)} val записей")
    
    text = extract_text(train + val)
    print(f"Общий текст: {len(text):,} символов")
    
    word_freq, bigram_freq = extract_terms(text)
    print(f"Уникальных слов: {len(word_freq):,}")
    print(f"Уникальных биграмм: {len(bigram_freq):,}")
    
    # Фильтруем
    candidates = filter_candidates(word_freq, bigram_freq, min_freq=5)
    
    # Выводим
    print("\n" + "="*60)
    print("АББРЕВИАТУРЫ (топ-50)")
    print("="*60)
    for term, freq in candidates["abbreviations"][:50]:
        print(f"  {term:20} — {freq}")
    
    print("\n" + "="*60)
    print("КАЗАХСКИЕ ТЕРМИНЫ (топ-100)")
    print("="*60)
    for term, freq in candidates["kazakh_specific"][:100]:
        print(f"  {term:30} — {freq}")
    
    print("\n" + "="*60)
    print("ЮРИДИЧЕСКИЕ ТЕРМИНЫ (1 слово, топ-50)")
    print("="*60)
    for term, freq in candidates["legal_unigrams"][:50]:
        print(f"  {term:30} — {freq}")
    
    print("\n" + "="*60)
    print("ЮРИДИЧЕСКИЕ БИГРАММЫ (топ-100)")
    print("="*60)
    for term, freq in candidates["legal_bigrams"][:100]:
        print(f"  {term:40} — {freq}")
    
    # Сохраняем всё в файл для экспертов
    output = {
        "abbreviations": candidates["abbreviations"][:100],
        "kazakh_terms": candidates["kazakh_specific"][:200],
        "legal_terms": candidates["legal_unigrams"][:100],
        "legal_bigrams": candidates["legal_bigrams"][:200],
    }
    
    with open("terms_for_experts.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    # Плоский список для быстрого просмотра
    with open("terms_for_experts.txt", "w", encoding="utf-8") as f:
        f.write("ТЕРМИНЫ ДЛЯ ПЕРЕВОДА ЭКСПЕРТАМИ\n")
        f.write("="*60 + "\n\n")
        
        f.write("АББРЕВИАТУРЫ:\n")
        for term, freq in candidates["abbreviations"][:100]:
            f.write(f"  {term} ({freq})\n")
        
        f.write("\nКАЗАХСКИЕ ТЕРМИНЫ:\n")
        for term, freq in candidates["kazakh_specific"][:200]:
            f.write(f"  {term} ({freq})\n")
        
        f.write("\nЮРИДИЧЕСКИЕ БИГРАММЫ:\n")
        for term, freq in candidates["legal_bigrams"][:200]:
            f.write(f"  {term} ({freq})\n")
    
    print(f"\nСохранено: terms_for_experts.json и terms_for_experts.txt")
    
    total = (len(candidates["abbreviations"][:100]) + 
             len(candidates["kazakh_specific"][:200]) + 
             len(candidates["legal_bigrams"][:200]))
    print(f"Всего терминов для проверки: ~{total}")

if __name__ == "__main__":
    main()

