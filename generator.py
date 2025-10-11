# generator.py
import os
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from api_handler import hf_generate, gemini_generate

LOCAL_MODEL = os.getenv('LOCAL_GENERATOR_MODEL', 'google/flan-t5-base')
USE_LOCAL = os.getenv('USE_LOCAL_GENERATOR', 'true').lower() in ('1','true','yes')

class Generator:
    def __init__(self):
        self.local_pipeline = None
        if USE_LOCAL:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(LOCAL_MODEL)
                # device -1 for CPU, 0 for GPU
                device = 0 if self._has_cuda() else -1
                self.local_pipeline = pipeline('text2text-generation', model=self.model, tokenizer=self.tokenizer, device=device)
            except Exception as e:
                print(f'[generator] local model load failed: {e}')
                self.local_pipeline = None

    def _has_cuda(self):
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False

    def generate(self, query: str, context: str, max_new_tokens: int = 200):
        prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
        if self.local_pipeline:
            try:
                out = self.local_pipeline(prompt, max_new_tokens=max_new_tokens)
                return out[0].get('generated_text') or out[0].get('text') or str(out)
            except Exception as e:
                print('[generator] local generation error:', e)

        hf_key = os.getenv('HF_API_KEY')
        if hf_key:
            try:
                return hf_generate(prompt, hf_key, max_new_tokens)
            except Exception as e:
                print('[generator] HF API error:', e)

        gem_key = os.getenv('GEMINI_API_KEY')
        if gem_key:
            try:
                return gemini_generate(prompt, gem_key)
            except Exception as e:
                print('[generator] Gemini API error:', e)

        return '⚠️ Unable to generate a response. No generator available.'
