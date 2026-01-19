"""
Live Chat с обученной Qwen3-14B (16-bit) моделью
Казахстанское законодательство
"""
import torch
from unsloth import FastLanguageModel
from transformers import TextStreamer

# ============== КОНФИГУРАЦИЯ ==============
MODEL_PATH = "outputs_14b_16bit/checkpoint-21303"
MAX_SEQ_LENGTH = 2048
LOAD_IN_4BIT = True  # Для инференса можно в 4-bit (экономит память)

# Prompt template (Alpaca format)
ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

def load_model():
    """Загрузка модели"""
    print("=" * 60)
    print("🚀 Загрузка Qwen3-14B (fine-tuned для КЗ законодательства)...")
    print("=" * 60)
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=LOAD_IN_4BIT,
    )
    FastLanguageModel.for_inference(model)
    
    # GPU info
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        print(f"✅ Модель загружена | GPU память: {allocated:.1f} GB")
    
    return model, tokenizer

def generate_response(model, tokenizer, instruction: str, input_text: str = "", 
                     max_tokens: int = 1024, temperature: float = 0.7, 
                     top_p: float = 0.9, stream: bool = True):
    """Генерация ответа"""
    prompt = ALPACA_PROMPT.format(instruction, input_text, "")
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    
    if stream:
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        output = model.generate(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_tokens,
            use_cache=True,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    else:
        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            use_cache=True,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        # Извлекаем только ответ
        if "### Response:" in response:
            response = response.split("### Response:")[-1].strip()
        print(response)
    
    return output

def run_tests(model, tokenizer):
    """Запуск тестовых примеров"""
    tests = [
        {
            "title": "Договор ГПХ",
            "instruction": "Ответь на вопрос по казахстанскому законодательству.",
            "input": "Можно ли заключить договор ГПХ с указанием цены работы за сутки?"
        },
        {
            "title": "Земельное право",
            "instruction": "Как регулируется обращение взыскания на земельные участки в РК?",
            "input": ""
        },
        {
            "title": "Трудовое право",
            "instruction": "Ответь на вопрос по казахстанскому законодательству.",
            "input": "Имеет ли право работник находиться на больничном более 60 дней?"
        },
        {
            "title": "Налоговое право", 
            "instruction": "Какие права имеют проверяемые лица при проведении налоговой проверки?",
            "input": ""
        },
    ]
    
    for i, test in enumerate(tests, 1):
        print(f"\n{'='*60}")
        print(f"📋 ТЕСТ {i}: {test['title']}")
        print(f"{'='*60}")
        print(f"❓ Instruction: {test['instruction']}")
        if test['input']:
            print(f"📥 Input: {test['input']}")
        print(f"\n💬 Response:")
        print("-" * 40)
        generate_response(model, tokenizer, test['instruction'], test['input'])
        print("-" * 40)

def chat_mode(model, tokenizer):
    """Интерактивный чат"""
    print("\n" + "=" * 60)
    print("💬 ИНТЕРАКТИВНЫЙ ЧАТ")
    print("=" * 60)
    print("Команды:")
    print("  exit, quit, q  - выход")
    print("  clear, cls     - очистить экран")
    print("  test           - запустить тесты")
    print("  temp <0.1-2.0> - изменить temperature")
    print("=" * 60)
    
    temperature = 0.7
    
    while True:
        print()
        try:
            user_input = input("🧑 Вы: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 До свидания!")
            break
        
        if not user_input:
            continue
        
        # Команды
        if user_input.lower() in ['exit', 'quit', 'q']:
            print("👋 До свидания!")
            break
        
        if user_input.lower() in ['clear', 'cls']:
            print("\033[2J\033[H", end="")  # Clear terminal
            continue
        
        if user_input.lower() == 'test':
            run_tests(model, tokenizer)
            continue
        
        if user_input.lower().startswith('temp '):
            try:
                temperature = float(user_input.split()[1])
                temperature = max(0.1, min(2.0, temperature))
                print(f"✅ Temperature установлен: {temperature}")
            except:
                print("❌ Использование: temp <0.1-2.0>")
            continue
        
        # Генерация ответа
        print(f"\n🤖 Ассистент (temp={temperature}):")
        print("-" * 40)
        generate_response(
            model, tokenizer,
            instruction="Ответь на вопрос по казахстанскому законодательству.",
            input_text=user_input,
            temperature=temperature
        )
        print("-" * 40)

def main():
    model, tokenizer = load_model()
    
    # Выбор режима
    print("\n" + "=" * 60)
    print("Выберите режим:")
    print("  1. Запустить тесты")
    print("  2. Интерактивный чат")
    print("  3. Оба (тесты + чат)")
    print("=" * 60)
    
    try:
        choice = input("Выбор [3]: ").strip() or "3"
    except (KeyboardInterrupt, EOFError):
        choice = "3"
    
    if choice in ["1", "3"]:
        run_tests(model, tokenizer)
    
    if choice in ["2", "3"]:
        chat_mode(model, tokenizer)

if __name__ == "__main__":
    main()
