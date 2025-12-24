"""
Класс-переводчик на базе NLLB-200
"""
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from pathlib import Path
from config import TRANSLATION_MODEL, LANG_CODES, MAX_LENGTH, CACHE_DIR


class Translator:
    def __init__(self, model_name: str = TRANSLATION_MODEL, device: str = "cuda"):
        print(f"Загрузка модели {model_name}...")
        self.device = device
        
        # Создаём кеш директорию если нужно
        cache_path = str(CACHE_DIR) if CACHE_DIR else None
        if cache_path:
            Path(cache_path).mkdir(parents=True, exist_ok=True)
            print(f"Кеш моделей: {cache_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            cache_dir=cache_path
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            cache_dir=cache_path
        ).to(device)
        self.model.eval()
        print("Модель загружена!")
    
    def detect_language(self, text: str) -> str:
        """
        Определение языка по специфичным казахским буквам.
        Казахский использует кириллицу + специальные буквы: Ә, Ғ, Қ, Ң, Ө, Ұ, Ү, Һ, І
        """
        if not text:
            return "ru"
        
        kk_specific = set("әғқңөұүһіӘҒҚҢӨҰҮҺІ")
        text_chars = set(text)
        kk_count = len(text_chars & kk_specific)
        
        # Если есть хотя бы 1 специфичная казахская буква - это казахский
        return "kk" if kk_count >= 1 else "ru"
    
    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """Перевод одного текста"""
        if not text or not text.strip():
            return ""
        
        self.tokenizer.src_lang = LANG_CODES[src_lang]
        
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=MAX_LENGTH
        ).to(self.device)
        
        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(LANG_CODES[tgt_lang]),
                max_new_tokens=MAX_LENGTH,
                num_beams=4,
                early_stopping=True,
            )
        
        return self.tokenizer.decode(generated[0], skip_special_tokens=True)
    
    def translate_batch(self, texts: list, src_lang: str, tgt_lang: str) -> list:
        """Перевод батча текстов"""
        if not texts:
            return []
        
        # Фильтруем пустые
        non_empty_indices = [i for i, t in enumerate(texts) if t and t.strip()]
        non_empty_texts = [texts[i] for i in non_empty_indices]
        
        if not non_empty_texts:
            return [""] * len(texts)
        
        self.tokenizer.src_lang = LANG_CODES[src_lang]
        
        inputs = self.tokenizer(
            non_empty_texts,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
        ).to(self.device)
        
        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(LANG_CODES[tgt_lang]),
                max_new_tokens=MAX_LENGTH,
                num_beams=4,
                early_stopping=True,
            )
        
        translated = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        
        # Восстанавливаем пустые
        result = [""] * len(texts)
        for idx, trans in zip(non_empty_indices, translated):
            result[idx] = trans
        
        return result
    
    def translate_record(self, record: dict) -> dict:
        """Перевод одной записи датасета в обратном направлении"""
        # Определяем язык по самому длинному полю
        sample = record.get("output", "") or record.get("instruction", "")
        src_lang = self.detect_language(sample)
        tgt_lang = "ru" if src_lang == "kk" else "kk"
        
        return {
            "instruction": self.translate(record.get("instruction", ""), src_lang, tgt_lang),
            "input": self.translate(record.get("input", ""), src_lang, tgt_lang),
            "output": self.translate(record.get("output", ""), src_lang, tgt_lang),
            "_original_lang": src_lang,
            "_translated_to": tgt_lang,
        }

