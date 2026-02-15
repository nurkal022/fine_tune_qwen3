"""
Интерактивный чат с обученной моделью
"""
import argparse
from unsloth import FastLanguageModel
from transformers import TextStreamer

from config import ALPACA_PROMPT

MAX_SEQ_LENGTH = 2048


def main():
    parser = argparse.ArgumentParser(description='Chat with fine-tuned model')
    parser.add_argument('--model', type=str, default='lora_qwen3_8b', help='Path to model')
    args = parser.parse_args()

    print("=" * 60)
    print(f"Loading model: {args.model}")
    print("=" * 60)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    print("Model loaded!\n")

    print("=" * 60)
    print("INTERACTIVE CHAT")
    print("Type 'exit' or 'quit' to leave")
    print("=" * 60)

    streamer = TextStreamer(tokenizer, skip_prompt=True)

    while True:
        print("\n" + "-" * 60)
        try:
            instruction = input("Question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if instruction.lower() in ['exit', 'quit']:
            print("Bye!")
            break

        if not instruction:
            continue

        input_text = input("Context (Enter to skip): ").strip()

        prompt = ALPACA_PROMPT.format(instruction, input_text, "")
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

        print("\nResponse:")
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
