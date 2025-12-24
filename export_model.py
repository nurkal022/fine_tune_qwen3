"""
Экспорт модели в различные форматы (16bit, GGUF)
"""
import argparse
from unsloth import FastLanguageModel
from config import ModelConfig

def load_model(model_path: str = "finetuned_qwen3_14b"):
    """Загрузка модели"""
    model_cfg = ModelConfig()
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=model_cfg.max_seq_length,
        dtype=model_cfg.dtype,
        load_in_4bit=model_cfg.load_in_4bit,
    )
    return model, tokenizer

def export_merged_16bit(model, tokenizer, output_dir: str = "model_16bit"):
    """Экспорт в float16 (merged)"""
    print(f"Экспорт в float16: {output_dir}")
    model.save_pretrained_merged(output_dir, tokenizer, save_method="merged_16bit")
    print("Готово!")

def export_merged_4bit(model, tokenizer, output_dir: str = "model_4bit"):
    """Экспорт в int4 (merged)"""
    print(f"Экспорт в int4: {output_dir}")
    model.save_pretrained_merged(output_dir, tokenizer, save_method="merged_4bit")
    print("Готово!")

def export_gguf(model, tokenizer, output_dir: str = "model_gguf", 
                quantization: str = "q8_0"):
    """Экспорт в GGUF формат"""
    print(f"Экспорт в GGUF ({quantization}): {output_dir}")
    model.save_pretrained_gguf(output_dir, tokenizer, quantization_method=quantization)
    print("Готово!")

def export_gguf_multiple(model, tokenizer, output_dir: str = "model_gguf"):
    """Экспорт в несколько GGUF форматов"""
    quantizations = ["q4_k_m", "q5_k_m", "q8_0"]
    print(f"Экспорт в GGUF (форматы: {quantizations}): {output_dir}")
    model.save_pretrained_gguf(
        output_dir, 
        tokenizer, 
        quantization_method=quantizations
    )
    print("Готово!")

def main():
    parser = argparse.ArgumentParser(description='Экспорт модели')
    parser.add_argument('--model', type=str, default='finetuned_qwen3_14b',
                        help='Путь к модели')
    parser.add_argument('--format', type=str, 
                        choices=['16bit', '4bit', 'gguf', 'gguf_all'],
                        required=True, help='Формат экспорта')
    parser.add_argument('--output', type=str, default=None,
                        help='Выходная директория')
    parser.add_argument('--quant', type=str, default='q8_0',
                        choices=['q4_k_m', 'q5_k_m', 'q8_0', 'f16'],
                        help='Тип квантизации для GGUF')
    args = parser.parse_args()
    
    print("Загрузка модели...")
    model, tokenizer = load_model(args.model)
    
    output = args.output
    
    if args.format == '16bit':
        output = output or "model_16bit"
        export_merged_16bit(model, tokenizer, output)
    elif args.format == '4bit':
        output = output or "model_4bit"
        export_merged_4bit(model, tokenizer, output)
    elif args.format == 'gguf':
        output = output or "model_gguf"
        export_gguf(model, tokenizer, output, args.quant)
    elif args.format == 'gguf_all':
        output = output or "model_gguf"
        export_gguf_multiple(model, tokenizer, output)

if __name__ == "__main__":
    main()

