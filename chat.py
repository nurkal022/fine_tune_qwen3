"""
Интерактивный чат с обученной моделью Qwen3-8B
"""
from unsloth import FastLanguageModel
from transformers import TextStreamer

MODEL_PATH = "finetuned_qwen3_8b"
MAX_SEQ_LENGTH = 2048

ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

def main():
    print("=" * 60)
    print("Загрузка модели...")
    print("=" * 60)
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    
    print("✅ Модель загружена!\n")
    
    print("=" * 60)
    print("ИНТЕРАКТИВНЫЙ ЧАТ")
    print("Введите 'exit' или 'quit' для выхода")
    print("=" * 60)
    
    streamer = TextStreamer(tokenizer, skip_prompt=True)
    
    while True:
        print("\n" + "-" * 60)
        instruction = input("💬 Ваш вопрос: ").strip()
        
        if instruction.lower() in ['exit', 'quit', 'выход']:
            print("\nДо свидания!")
            break
        
        if not instruction:
            continue
        
        input_text = input("📝 Контекст (Enter для пропуска): ").strip()
        
        prompt = ALPACA_PROMPT.format(instruction, input_text, "")
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
        
        print("\n🤖 Ответ:")
        print("-" * 60)
        _ = model.generate(
            **inputs,
            streamer=streamer,
            max_new_tokens=512,
            use_cache=True,
            temperature=0.7,
            top_p=0.9,
        )
        print("-" * 60)

if __name__ == "__main__":
    main()

