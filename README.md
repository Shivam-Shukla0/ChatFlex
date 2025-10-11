# NYD2026_ChatFlex

Advanced Retrieval-Augmented Generation (RAG) chatbot built for NYD Hackathon 2026.
Combines FAISS retrieval, SentenceTransformers embeddings, a local open-source LLM
generator (Flan-T5 by default) and optional API fallbacks (Hugging Face / Google Gemini).

## Features
- Upload any TXT/CSV/JSON dataset (no code changes needed)
- Chunking + SentenceTransformers embeddings
- FAISS vector index for fast retrieval
- Local generator (Flan-T5) with optional API fallback to HuggingFace or Gemini
- Flask UI with multi-dataset support and source snippets
- Configurable via .env

## Quickstart (Local CPU)
1. Create & activate a virtualenv
   ```bash
   python -m venv venv
   source venv/bin/activate   # on Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```
2. Copy `.env.example` to `.env` and edit if you want API fallbacks.
3. Run:
   ```bash
   python app.py
   ```
4. Open `http://127.0.0.1:5000/`

## Notes
- By default the project uses `google/flan-t5-base` as local generator. On CPU this is slower but usable for demos.
- If you have a GPU, install the GPU build of PyTorch and the model will use GPU automatically.
- For Gemini integration, set `GEMINI_API_KEY` in `.env` and follow official Google Generative AI SDK setup.

## Structure
- app.py - Flask app and endpoints
- rag_pipeline.py - chunking, embedding, and FAISS index management
- generator.py - local generator + API fallback
- api_handler.py - wrappers for HF/Gemini calls (Gemini stub uses google-generativeai if configured)
- templates/index.html - simple web UI

## License
MIT
