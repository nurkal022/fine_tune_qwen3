"""
Тест переводчика на нескольких примерах
"""
from translator import Translator


def main():
    print("="*50)
    print("ТЕСТ ПЕРЕВОДЧИКА")
    print("="*50)
    
    translator = Translator()
    
    # Тестовые примеры
    examples = [
        {
            "instruction": "Что означает НДС?",
            "input": "",
            "output": "Налог на добавленную стоимость."
        },
        {
            "instruction": "Жерді пайдалану құқығын уақытша беру нені білдіреді?",
            "input": "",
            "output": "Бұл белгілі бір мерзімге жер беру."
        },
        {
            "instruction": "Ответь на вопрос по казахстанскому законодательству.",
            "input": "Можно ли заключить договор ГПХ с указанием цены работы за сутки?",
            "output": "Да, это допустимо согласно Гражданскому кодексу РК."
        },
    ]
    
    for i, ex in enumerate(examples, 1):
        print(f"\n{'='*50}")
        print(f"ПРИМЕР {i}")
        print(f"{'='*50}")
        
        # Определяем язык
        sample = ex["output"] or ex["instruction"]
        src_lang = translator.detect_language(sample)
        tgt_lang = "ru" if src_lang == "kk" else "kk"
        
        print(f"\nОригинал ({src_lang}):")
        print(f"  instruction: {ex['instruction']}")
        print(f"  input: {ex['input']}")
        print(f"  output: {ex['output']}")
        
        # Переводим
        translated = translator.translate_record(ex)
        
        print(f"\nПеревод ({tgt_lang}):")
        print(f"  instruction: {translated['instruction']}")
        print(f"  input: {translated['input']}")
        print(f"  output: {translated['output']}")
    
    print(f"\n{'='*50}")
    print("ТЕСТ ЗАВЕРШЁН")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()

