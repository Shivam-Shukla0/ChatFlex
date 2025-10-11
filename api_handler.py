# api_handler.py
import os
import requests

def hf_generate(prompt, hf_key, max_new_tokens=200, model='google/flan-t5-base'):
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {'Authorization': f'Bearer {hf_key}'}
    payload = {'inputs': prompt, 'parameters': {'max_new_tokens': max_new_tokens}}
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    out = r.json()
    if isinstance(out, list):
        return out[0].get('generated_text', str(out))
    elif isinstance(out, dict):
        return out.get('generated_text') or str(out)
    return str(out)

def gemini_generate(prompt, gemini_key, model='gemini-1.5-mini'):
    # Use the official google-generativeai SDK for production usage.
    # This is a minimal placeholder if you prefer not to install the SDK.
    raise NotImplementedError('Use google.generativeai SDK for Gemini calls. See README.')
