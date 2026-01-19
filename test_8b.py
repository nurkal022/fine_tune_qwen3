"""
Тест обученной Qwen3-8B модели
"""
from unsloth import FastLanguageModel
from transformers import TextStreamer

MODEL_PATH = "outputs_14b_16bit/checkpoint-21303"
MAX_SEQ_LENGTH = 2048

ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

print("Загрузка модели...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)
print("Готово!\n")

def ask(instruction: str, input_text: str = "", max_tokens: int = 512):
    """Генерация ответа"""
    prompt = ALPACA_PROMPT.format(instruction, input_text, "")
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    
    print(f"📝 Instruction: {instruction}")
    if input_text:
        print(f"📥 Input: {input_text}")
    print("\n💬 Response:")
    print("-" * 40)
    
    streamer = TextStreamer(tokenizer, skip_prompt=True)
    _ = model.generate(
        **inputs,
        streamer=streamer,
        max_new_tokens=max_tokens,
        use_cache=True,
        temperature=0.7,
        top_p=0.9,
    )
    print("-" * 40)

# Тестовые примеры
print("=" * 50)
print("ТЕСТ 1: Общий вопрос по законодательству")
print("=" * 50)
ask(
    instruction="Ответь на вопрос по казахстанскому законодательству.",
    input_text="Можно ли заключить договор ГПХ с указанием цены работы за сутки?"
)

print("\n" + "=" * 50)
print("ТЕСТ 2: Земельное право")
print("=" * 50)
ask(
    instruction="Как регулируется обращение взыскания на земельные участки?",
    input_text=""
)

print("\n" + "=" * 50)
print("ТЕСТ 3: Трудовое право")
print("=" * 50)
ask(
    instruction="Ответь на вопрос по казахстанскому законодательству.",
    input_text="Имеет ли право работник находиться на больничном более 60 дней?"
)

print("\n" + "=" * 50)
print("ТЕСТ 4: Налоговое право")
print("=" * 50)
ask(
    instruction="Какие права имеют проверяемые лица при проведении налоговой проверки?",
    input_text=""
)

print("\n" + "=" * 50)
print("ИНТЕРАКТИВНЫЙ РЕЖИМ")
print("Введите 'exit' для выхода")
print("=" * 50)

while True:
    print()
    instruction = input("Instruction: ").strip()
    if instruction.lower() == 'exit':
        break
    input_text = input("Input (Enter для пустого): ").strip()
    print()
    ask(instruction, input_text)

