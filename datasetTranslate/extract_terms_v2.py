"""
Продвинутое извлечение юридических терминов
TF-IDF + морфология + паттерны
"""
import json
import re
import math
from collections import Counter, defaultdict
from pathlib import Path

# Казахские специфичные буквы
KK_CHARS = set("әғқңөұүһіӘҒҚҢӨҰҮҺІ")

# Расширенные стоп-слова (местоимения, союзы, частицы, служебные)
STOPWORDS_RU = {
    # Местоимения
    'я', 'ты', 'он', 'она', 'оно', 'мы', 'вы', 'они',
    'мой', 'твой', 'его', 'её', 'ее', 'наш', 'ваш', 'их',
    'этот', 'тот', 'такой', 'какой', 'который', 'чей',
    'сам', 'самый', 'весь', 'всё', 'все', 'каждый', 'любой',
    'меня', 'тебя', 'себя', 'нас', 'вас', 'мне', 'тебе', 'ему', 'ей', 'нам', 'вам', 'им',
    'эта', 'это', 'эти', 'того', 'этого', 'этой', 'этих', 'тех', 'той', 'тому',
    # Союзы и частицы  
    'и', 'а', 'но', 'или', 'да', 'ни', 'то', 'не', 'же', 'ли', 'бы',
    'что', 'как', 'когда', 'где', 'куда', 'откуда', 'почему', 'зачем',
    'если', 'чтобы', 'хотя', 'пока', 'либо', 'также', 'тоже', 'даже',
    'ведь', 'вот', 'вон', 'уже', 'ещё', 'еще', 'лишь', 'только', 'именно',
    # Предлоги
    'в', 'на', 'с', 'со', 'к', 'по', 'за', 'из', 'от', 'до', 'у', 'о', 'об', 'обо',
    'при', 'для', 'без', 'под', 'над', 'между', 'через', 'перед', 'после',
    # Глаголы-связки и частые глаголы
    'быть', 'есть', 'был', 'была', 'было', 'были', 'будет', 'будут',
    'является', 'являются', 'являлся', 'являлись',
    'может', 'могут', 'мог', 'могла', 'могли', 'можно', 'нельзя',
    'должен', 'должна', 'должно', 'должны', 'нужно', 'надо',
    'имеет', 'имеют', 'имел', 'имела', 'имели',
    'следует', 'стоит', 'необходимо',
    # Наречия
    'так', 'там', 'тут', 'здесь', 'очень', 'более', 'менее', 'наиболее',
    'всегда', 'никогда', 'иногда', 'часто', 'редко', 'уже', 'ещё',
    'сейчас', 'теперь', 'потом', 'затем', 'далее', 'выше', 'ниже',
    # Числительные
    'один', 'два', 'три', 'четыре', 'пять', 'первый', 'второй', 'третий',
    # Прочее частое
    'год', 'года', 'году', 'годы', 'лет', 'день', 'дня', 'дней', 'дни',
    'раз', 'раза', 'том', 'число', 'числе', 'время', 'период',
    'случай', 'случае', 'случаях', 'образ', 'образом',
    'основание', 'основании', 'соответствие', 'соответствии',
    'течение', 'рамках', 'силу', 'связи', 'зависимости', 'качестве',
    'отношение', 'отношении', 'виде', 'порядке', 'размере',
    'данный', 'данном', 'данного', 'данной', 'данных', 'данные',
    'настоящий', 'настоящего', 'настоящей', 'настоящим', 'настоящее',
    'указанный', 'указанного', 'указанной', 'указанных', 'указанные',
    'другой', 'другого', 'других', 'другие', 'иной', 'иного', 'иных', 'иные',
    # Шаблоны из датасета
    'ответь', 'вопрос', 'здравствуйте', 'добрый', 'подскажите', 'пожалуйста',
    'спасибо', 'благодарю', 'заранее',
}

STOPWORDS_KK = {
    # Местоимения
    'мен', 'сен', 'ол', 'біз', 'сіз', 'олар', 'бұл', 'осы', 'сол', 'ол',
    'менің', 'сенің', 'оның', 'біздің', 'сіздің', 'олардың',
    'маған', 'саған', 'оған', 'бізге', 'сізге', 'оларға',
    'мені', 'сені', 'оны', 'бізді', 'сізді', 'оларды',
    'қандай', 'қалай', 'қашан', 'қайда', 'кім', 'не', 'нені', 'неге', 'неліктен',
    'қай', 'қайсы', 'неше', 'қанша',
    # Союзы, частицы
    'және', 'немесе', 'бірақ', 'сондай', 'сондықтан', 'өйткені', 'себебі',
    'да', 'де', 'та', 'те', 'ма', 'ме', 'ба', 'бе', 'па', 'пе',
    'ғой', 'ғана', 'тек', 'ғой', 'шығар',
    # Послелоги
    'үшін', 'туралы', 'бойынша', 'арқылы', 'қарай', 'дейін', 'кейін',
    'бұрын', 'соң', 'басқа', 'сияқты', 'секілді', 'сайын',
    # Глаголы-связки
    'бар', 'жоқ', 'болу', 'болады', 'болып', 'болған', 'болса', 'болмайды',
    'ету', 'етеді', 'етіп', 'еткен', 'етілді', 'етіледі',
    'келеді', 'келіп', 'алады', 'алып', 'береді', 'беріп',
    'мүмкін', 'керек', 'тиіс', 'қажет',
    # Наречия
    'өте', 'тым', 'ең', 'анағұрлым', 'барлық', 'бірге',
    # Шаблоны
    'жауап', 'сұрақ',
}

def load_jsonl(path):
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records

def has_kk_chars(word):
    return bool(set(word) & KK_CHARS)

def is_abbreviation(word):
    """Аббревиатуры: ТОО, РК, ЖШС, КоАП"""
    if len(word) < 2 or len(word) > 10:
        return False
    if word.isupper() and len(word) >= 2:
        return True
    upper_count = sum(1 for c in word if c.isupper())
    if upper_count >= 2 and len(word) <= 6:
        return True
    return False

def is_legal_noun_ru(word):
    """Проверяем, похоже ли на юридическое существительное (русский)"""
    word_lower = word.lower()
    
    # Типичные суффиксы существительных
    noun_suffixes = [
        'ость', 'ение', 'ание', 'ство', 'тель', 'ция', 'сия',
        'тор', 'ент', 'ант', 'изм', 'ист', 'ика', 'логия',
        'ёр', 'ер', 'арь', 'яр', 'ник', 'чик', 'щик',
    ]
    
    # Юридические корни
    legal_roots = [
        'закон', 'право', 'суд', 'кодекс', 'статья', 'пункт',
        'договор', 'контракт', 'сделк', 'обязательств',
        'ответствен', 'полномоч', 'компетенц',
        'орган', 'учреждени', 'организац',
        'налог', 'бюджет', 'финанс', 'имуществ', 'собственн',
        'регистрац', 'лицензи', 'разрешени', 'сертификат',
        'истец', 'ответчик', 'заявител', 'должник', 'кредитор',
        'нотариус', 'адвокат', 'прокурор', 'судья', 'следовател',
        'правонаруш', 'преступлен', 'штраф', 'санкц', 'взыскан',
        'иск', 'жалоб', 'апелляц', 'кассац', 'надзор',
        'наследств', 'завещан', 'дарен', 'аренд', 'найм',
        'акционер', 'учредител', 'участник', 'директор',
        'работник', 'работодател', 'трудов',
        'маслихат', 'аким', 'акимат',
    ]
    
    for suffix in noun_suffixes:
        if word_lower.endswith(suffix) and len(word_lower) > len(suffix) + 2:
            return True
    
    for root in legal_roots:
        if root in word_lower:
            return True
    
    return False

def is_legal_noun_kk(word):
    """Проверяем, похоже ли на юридический термин (казахский)"""
    word_lower = word.lower()
    
    # Юридические корни на казахском
    legal_roots = [
        'заң', 'құқық', 'сот', 'кодекс', 'бап', 'тарма',
        'шарт', 'келісім', 'міндеттем',
        'жауапкершілік', 'өкілеттік',
        'орган', 'мекеме', 'ұйым',
        'салық', 'бюджет', 'қаржы', 'мүлік', 'меншік',
        'тіркеу', 'лицензия', 'рұқсат', 'куәлік',
        'талапкер', 'жауапкер', 'өтініш', 'борышкер', 'кредитор',
        'нотариус', 'адвокат', 'прокурор', 'судья', 'тергеуші',
        'құқықбұзушылық', 'қылмыс', 'айыппұл', 'санкция', 'өндіріп',
        'талап', 'шағым', 'апелляция', 'кассация', 'қадағалау',
        'мұра', 'өсиет', 'сыйлық', 'жалға', 'жалдау',
        'акционер', 'құрылтайшы', 'қатысушы', 'директор',
        'жұмысшы', 'жұмыс беруші', 'еңбек',
        'мәслихат', 'әкім', 'әкімшілік', 'әкімдік',
        'Жоғарғы', 'облыстық', 'аудандық',
        'конституция', 'республика',
    ]
    
    for root in legal_roots:
        if root in word_lower:
            return True
    
    return False

def extract_documents(records):
    """Каждая запись = документ для TF-IDF"""
    docs = []
    for r in records:
        text = f"{r.get('instruction', '')} {r.get('input', '')} {r.get('output', '')}"
        docs.append(text)
    return docs

def tokenize(text):
    """Токенизация кириллицы"""
    return re.findall(r'[А-Яа-яӘәҒғҚқҢңӨөҰұҮүҺһІі]+', text)

def compute_tfidf(docs, min_df=5, max_df_ratio=0.3):
    """
    Вычисляем TF-IDF для извлечения терминов
    min_df - минимум документов с термином
    max_df_ratio - максимальная доля документов (исключаем слишком частые)
    """
    # Document frequency
    df = Counter()
    # Term frequency per doc
    tf_docs = []
    
    for doc in docs:
        tokens = tokenize(doc)
        tf = Counter(tokens)
        tf_docs.append(tf)
        for token in set(tokens):
            df[token] += 1
    
    n_docs = len(docs)
    max_df = int(n_docs * max_df_ratio)
    
    # Фильтруем по DF
    valid_terms = {
        term for term, count in df.items()
        if min_df <= count <= max_df
    }
    
    # Вычисляем IDF
    idf = {}
    for term in valid_terms:
        idf[term] = math.log(n_docs / (df[term] + 1)) + 1
    
    # Агрегируем TF-IDF по всему корпусу
    tfidf_scores = defaultdict(float)
    for tf in tf_docs:
        for term, count in tf.items():
            if term in valid_terms:
                tfidf_scores[term] += count * idf[term]
    
    return dict(tfidf_scores), df

def extract_ngrams(docs, n=2, min_freq=10):
    """Извлекаем n-граммы"""
    ngram_freq = Counter()
    
    for doc in docs:
        tokens = tokenize(doc)
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            ngram_freq[ngram] += 1
    
    return {
        ' '.join(ng): freq 
        for ng, freq in ngram_freq.items() 
        if freq >= min_freq
    }

def is_good_bigram(bigram, stopwords_all):
    """Проверяем, что биграмма - не мусор"""
    words = bigram.split()
    if len(words) != 2:
        return False
    
    w1, w2 = words
    w1_low, w2_low = w1.lower(), w2.lower()
    
    # Оба слова в стоп-словах - мусор
    if w1_low in stopwords_all and w2_low in stopwords_all:
        return False
    
    # Первое слово короткое служебное
    if len(w1) <= 2 and w1_low in stopwords_all:
        return False
    
    # Оба слова слишком короткие
    if len(w1) <= 2 and len(w2) <= 2:
        return False
    
    # Хотя бы одно слово должно быть существенным (>4 букв, не стоп)
    has_substantial = False
    for w in [w1_low, w2_low]:
        if len(w) >= 5 and w not in stopwords_all:
            has_substantial = True
            break
    
    if not has_substantial:
        return False
    
    # Паттерны мусора из датасета
    junk_patterns = [
        'ответь на', 'на вопрос', 'вопрос по', 'по казахстанскому',
        'казахстанскому законодательству', 'законодательству как',
        'законодательству в', 'законодательству здравствуйте',
        'законодательству можно', 'законодательству может',
        'законодательству какие', 'законодательству если',
        'законодательству тоо', 'добрый день', 'можно ли',
        'может ли', 'нужно ли', 'имеет ли', 'как правильно',
        'таким образом', 'кроме того', 'при этом', 'в соответствии',
        'соответствии с', 'соответствии со', 'в случае', 'случае если',
        'в связи', 'связи с', 'в течение', 'в рамках', 'в размере',
        'в качестве', 'в виде', 'в порядке', 'в отношении',
        'на основании', 'исходя из', 'в зависимости', 'зависимости от',
        'том числе', 'данном случае', 'согласно п', 'согласно ст',
        'года о', 'года об', 'года далее', 'далее по', 'далее закон',
        'по тексту', 'рк далее', 'казахстан далее', 'по состоянию',
        'по месту', 'со стороны', 'за счет', 'за исключением',
        'не менее', 'не предусмотрено', 'должно быть',
        'вы можете', 'мы полагаем', 'полагаем что',
        'привести к', 'может привести', 'повлечь за', 'влечет за',
        'может повлечь', 'означает что', 'то есть',
        'от декабря', 'от января', 'от июля', 'от апреля',
        'декабря года', 'января года', 'июля года', 'апреля года',
        'потому что',
    ]
    
    if bigram.lower() in junk_patterns:
        return False
    
    return True

def filter_terms(tfidf_scores, df, stopwords_all):
    """Фильтруем термины"""
    
    result = {
        'abbreviations': [],
        'legal_terms_ru': [],
        'legal_terms_kk': [],
    }
    
    for term, score in tfidf_scores.items():
        term_lower = term.lower()
        freq = df.get(term, 0)
        
        # Пропускаем стоп-слова
        if term_lower in stopwords_all:
            continue
        
        # Слишком короткие
        if len(term) < 3:
            continue
        
        # Аббревиатуры
        if is_abbreviation(term):
            result['abbreviations'].append((term, score, freq))
            continue
        
        # Казахские термины
        if has_kk_chars(term):
            if is_legal_noun_kk(term) and len(term) >= 4:
                result['legal_terms_kk'].append((term, score, freq))
            continue
        
        # Русские юридические термины
        if is_legal_noun_ru(term) and len(term) >= 5:
            result['legal_terms_ru'].append((term, score, freq))
    
    # Сортируем по TF-IDF score
    for key in result:
        result[key] = sorted(result[key], key=lambda x: -x[1])
    
    return result

def filter_bigrams(bigrams, stopwords_all):
    """Фильтруем биграммы"""
    
    legal_bigram_patterns = [
        # Русские юридические паттерны
        (r'(трудов|гражданск|налогов|уголовн|административн|бюджетн|земельн)', r'(кодекс|право|договор|законодательств|ответственност)'),
        (r'(государственн|местн|исполнительн|представительн)', r'(орган|власт|управлен|регистрац|закупк)'),
        (r'(юридическ|физическ|должностн)', r'(лиц|ответственност)'),
        (r'(судебн|арбитражн|апелляционн|кассационн)', r'(порядок|инстанц|разбирательств|решен|акт)'),
        (r'(обязательн|социальн|пенсионн|медицинск)', r'(страхован|отчислен|взнос|выплат)'),
        (r'(заработн|среднемесячн)', r'(плат|оклад|зарплат)'),
        (r'(ограниченн|дополнительн|материальн|административн)', r'(ответственност)'),
        (r'(нормативн|правов)', r'(акт|база|регулирован)'),
        (r'(индивидуальн|частн)', r'(предпринимател)'),
    ]
    
    result = []
    
    for bigram, freq in bigrams.items():
        if not is_good_bigram(bigram, stopwords_all):
            continue
        
        words = bigram.split()
        w1, w2 = words[0].lower(), words[1].lower()
        
        # Проверяем юридические паттерны
        is_legal = False
        for pat1, pat2 in legal_bigram_patterns:
            if re.search(pat1, w1) and re.search(pat2, w2):
                is_legal = True
                break
            if re.search(pat1, w2) and re.search(pat2, w1):
                is_legal = True
                break
        
        # Или хотя бы одно слово - юридический термин
        if not is_legal:
            if is_legal_noun_ru(w1) or is_legal_noun_ru(w2):
                is_legal = True
            if is_legal_noun_kk(w1) or is_legal_noun_kk(w2):
                is_legal = True
        
        if is_legal:
            result.append((bigram, freq))
    
    return sorted(result, key=lambda x: -x[1])

def main():
    print("="*60)
    print("ИЗВЛЕЧЕНИЕ ЮРИДИЧЕСКИХ ТЕРМИНОВ v2")
    print("TF-IDF + морфология + паттерны")
    print("="*60)
    
    # Загрузка
    train = load_jsonl("combined_data/train.jsonl")
    val = load_jsonl("combined_data/validation.jsonl")
    all_records = train + val
    
    print(f"\nЗагружено: {len(all_records)} записей")
    
    # Документы для TF-IDF
    docs = extract_documents(all_records)
    
    # TF-IDF
    print("Вычисляем TF-IDF...")
    tfidf_scores, df = compute_tfidf(docs, min_df=10, max_df_ratio=0.25)
    print(f"Терминов после фильтрации DF: {len(tfidf_scores)}")
    
    # Объединённые стоп-слова
    stopwords_all = STOPWORDS_RU | STOPWORDS_KK
    
    # Фильтруем униграммы
    print("Фильтруем термины...")
    terms = filter_terms(tfidf_scores, df, stopwords_all)
    
    # Биграммы
    print("Извлекаем биграммы...")
    bigrams = extract_ngrams(docs, n=2, min_freq=20)
    legal_bigrams = filter_bigrams(bigrams, stopwords_all)
    
    # Триграммы (для составных терминов)
    print("Извлекаем триграммы...")
    trigrams = extract_ngrams(docs, n=3, min_freq=15)
    # Фильтруем триграммы простым способом
    legal_trigrams = []
    for trigram, freq in trigrams.items():
        words = trigram.split()
        # Хотя бы 2 слова должны быть не стоп-словами
        non_stop = sum(1 for w in words if w.lower() not in stopwords_all and len(w) >= 4)
        if non_stop >= 2:
            # Проверяем юридичность
            has_legal = any(is_legal_noun_ru(w) or is_legal_noun_kk(w) for w in words)
            if has_legal:
                legal_trigrams.append((trigram, freq))
    legal_trigrams = sorted(legal_trigrams, key=lambda x: -x[1])
    
    # Выводим результаты
    print("\n" + "="*60)
    print("АББРЕВИАТУРЫ (топ-50)")
    print("="*60)
    for term, score, freq in terms['abbreviations'][:50]:
        print(f"  {term:15} freq={freq:5}  tfidf={score:.1f}")
    
    print("\n" + "="*60)
    print("ЮРИДИЧЕСКИЕ ТЕРМИНЫ (русские, топ-100)")
    print("="*60)
    for term, score, freq in terms['legal_terms_ru'][:100]:
        print(f"  {term:30} freq={freq:5}  tfidf={score:.1f}")
    
    print("\n" + "="*60)
    print("ЮРИДИЧЕСКИЕ ТЕРМИНЫ (казахские, топ-100)")
    print("="*60)
    for term, score, freq in terms['legal_terms_kk'][:100]:
        print(f"  {term:30} freq={freq:5}  tfidf={score:.1f}")
    
    print("\n" + "="*60)
    print("ЮРИДИЧЕСКИЕ БИГРАММЫ (топ-100)")
    print("="*60)
    for bigram, freq in legal_bigrams[:100]:
        print(f"  {bigram:45} freq={freq}")
    
    print("\n" + "="*60)
    print("ЮРИДИЧЕСКИЕ ТРИГРАММЫ (топ-50)")
    print("="*60)
    for trigram, freq in legal_trigrams[:50]:
        print(f"  {trigram:55} freq={freq}")
    
    # Сохраняем
    output = {
        "abbreviations": [(t, int(f)) for t, s, f in terms['abbreviations'][:100]],
        "legal_terms_ru": [(t, int(f)) for t, s, f in terms['legal_terms_ru'][:200]],
        "legal_terms_kk": [(t, int(f)) for t, s, f in terms['legal_terms_kk'][:200]],
        "legal_bigrams": [(b, int(f)) for b, f in legal_bigrams[:200]],
        "legal_trigrams": [(t, int(f)) for t, f in legal_trigrams[:100]],
    }
    
    with open("terms_for_experts_v2.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    # Текстовый файл для экспертов
    with open("terms_for_experts_v2.txt", "w", encoding="utf-8") as f:
        f.write("ЮРИДИЧЕСКИЕ ТЕРМИНЫ ДЛЯ ГЛОССАРИЯ\n")
        f.write("="*60 + "\n")
        f.write("(отфильтровано TF-IDF + морфология)\n\n")
        
        f.write("АББРЕВИАТУРЫ:\n")
        f.write("-"*40 + "\n")
        for term, score, freq in terms['abbreviations'][:100]:
            f.write(f"  {term} ({freq})\n")
        
        f.write("\nРУССКИЕ ЮРИДИЧЕСКИЕ ТЕРМИНЫ:\n")
        f.write("-"*40 + "\n")
        for term, score, freq in terms['legal_terms_ru'][:200]:
            f.write(f"  {term} ({freq})\n")
        
        f.write("\nКАЗАХСКИЕ ЮРИДИЧЕСКИЕ ТЕРМИНЫ:\n")
        f.write("-"*40 + "\n")
        for term, score, freq in terms['legal_terms_kk'][:200]:
            f.write(f"  {term} ({freq})\n")
        
        f.write("\nЮРИДИЧЕСКИЕ СЛОВОСОЧЕТАНИЯ (2 слова):\n")
        f.write("-"*40 + "\n")
        for bigram, freq in legal_bigrams[:200]:
            f.write(f"  {bigram} ({freq})\n")
        
        f.write("\nЮРИДИЧЕСКИЕ СЛОВОСОЧЕТАНИЯ (3 слова):\n")
        f.write("-"*40 + "\n")
        for trigram, freq in legal_trigrams[:100]:
            f.write(f"  {trigram} ({freq})\n")
    
    total = (len(terms['abbreviations'][:100]) + 
             len(terms['legal_terms_ru'][:200]) + 
             len(terms['legal_terms_kk'][:200]) +
             len(legal_bigrams[:200]) +
             len(legal_trigrams[:100]))
    
    print(f"\nСохранено: terms_for_experts_v2.json и terms_for_experts_v2.txt")
    print(f"Всего терминов для проверки: ~{total}")

if __name__ == "__main__":
    main()

