"""
Инференс (тестирование) обученной модели
"""
import torch
from unsloth import FastLanguageModel
from transformers import TextStreamer

from config import ModelConfig, ALPACA_PROMPT

def load_finetuned_model(model_path: str = "finetuned_qwen3_14b"):
    """Загрузка дообученной модели"""
    print(f"Загрузка модели из {model_path}...")
    
    model_cfg = ModelConfig()
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=model_cfg.max_seq_length,
        dtype=model_cfg.dtype,
        load_in_4bit=model_cfg.load_in_4bit,
    )
    
    FastLanguageModel.for_inference(model)
    print("Модель загружена и готова к инференсу!")
    
    return model, tokenizer

def generate(model, tokenizer, instruction: str, input_text: str = "", 
             max_new_tokens: int = 512, stream: bool = True):
    """Генерация ответа"""
    prompt = ALPACA_PROMPT.format(instruction, input_text, "")
    
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    
    if stream:
        text_streamer = TextStreamer(tokenizer)
        outputs = model.generate(
            **inputs,
            streamer=text_streamer,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )
    else:
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )
        result = tokenizer.batch_decode(outputs)[0]
        return result
    
    return None

def interactive_mode(model, tokenizer):
    """Интерактивный режим"""
    print("\n" + "=" * 50)
    print("Интерактивный режим")
    print("Введите 'exit' для выхода")
    print("=" * 50)
    
    while True:
        print("\n")
        instruction = input("Instruction: ").strip()
        
        if instruction.lower() == 'exit':
            break
        
        input_text = input("Input (можно пустое): ").strip()
        
        print("\nResponse:")
        print("-" * 30)
        generate(model, tokenizer, instruction, input_text)
        print("-" * 30)

def test_examples(model, tokenizer):
    """Тестовые примеры"""
    examples = [
        {
            "instruction": "Кто имеет право пользоваться недрами в пределах участка?",
            "input": ""
        },
        {
            "instruction": "Ответь на вопрос по казахстанскому законодательству.",
            "input": "Можно ли заключить договор ГПХ только с указанием цены работы за сутки?"
        },
        {
            "instruction": "Как регулируется обращение взыскания на земельные участки?",
            "input": ""
        },
    ]
    
    print("\n" + "=" * 50)
    print("Тестовые примеры")
    print("=" * 50)
    
    for i, example in enumerate(examples, 1):
        print(f"\n--- Пример {i} ---")
        print(f"Instruction: {example['instruction']}")
        if example['input']:
            print(f"Input: {example['input']}")
        print("\nResponse:")
        generate(model, tokenizer, example['instruction'], example['input'])
        print()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Инференс Qwen3-14B')
    parser.add_argument('--model', type=str, default='finetuned_qwen3_14b',
                        help='Путь к модели')
    parser.add_argument('--mode', type=str, choices=['interactive', 'test'], 
                        default='test', help='Режим работы')
    args = parser.parse_args()
    
    model, tokenizer = load_finetuned_model(args.model)
    
    if args.mode == 'interactive':
        interactive_mode(model, tokenizer)
    else:
        test_examples(model, tokenizer)

if __name__ == "__main__":
    main()

