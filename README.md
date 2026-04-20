# ChatFlex

> **Instantly turn any dataset into a working chatbot — no code changes needed.**

ChatFlex is a lightweight, open-source RAG (Retrieval-Augmented Generation) chatbot that allows users to upload any text or CSV dataset and immediately start asking questions about it. Built for the **NYD 2026 Hackathon**, it combines vector-based semantic search with LLM-powered response generation for accurate, context-aware answers.

---

## What Makes It Different

Most chatbots are hardcoded for one domain. ChatFlex is **domain-agnostic** — upload a Ramayana dataset, a medical handbook, a product FAQ, or any CSV/text file, and it instantly becomes an expert on that topic. No retraining. No code changes.

---

## Features

- **Upload Any Dataset** — Supports `.txt` and `.csv` files up to 10MB+
- **Background Processing** — Large files are indexed in a background thread; the app stays responsive during upload
- **Semantic Search** — Vector store retrieves the most relevant context chunks for each query
- **LLM-Powered Answers** — Integrates with HuggingFace or Gemini API for intelligent response generation
- **Flask Web Interface** — Simple, clean browser-based UI for upload and Q&A
- **Plug-and-Play Dataset Switching** — Swap datasets at runtime without restarting the server

---

## How It Works

```
User uploads dataset (CSV / TXT)
        ↓
data_loader.py → splits content into chunks
        ↓
vector_store.py → creates semantic vector index
        ↓
User asks a question
        ↓
vector_store.py → finds top relevant chunks (context)
        ↓
chatbot.py → sends question + context to LLM API
        ↓
Answer displayed in browser
```

This is a classic **RAG (Retrieval-Augmented Generation)** pipeline — the same architecture used in enterprise AI assistants.

---

## Project Structure

```
ChatFlex/
├── app.py               # Flask server, routes, background threading
├── chatbot.py           # LLM API integration (HuggingFace / Gemini)
├── data_loader.py       # Text chunking and preprocessing
├── vector_store.py      # Vector index creation and semantic search
├── requirements.txt     # Python dependencies
├── sample_data/         # Example datasets to test with
└── templates/
    └── index.html       # Frontend UI
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| AI / LLM | HuggingFace API / Gemini API |
| Vector Search | Custom vector store (cosine similarity) |
| Text Processing | Chunking, tokenization |
| Frontend | HTML, CSS |

---

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/Shivam-Shukla0/ChatFlex.git
cd ChatFlex
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your API key
In `chatbot.py`, set your HuggingFace or Gemini API key:
```python
API_KEY = "your_api_key_here"
```

### 4. Run the app
```bash
python app.py
```

### 5. Open in browser
```
http://127.0.0.1:5000
```

---

## Example Use Cases

- **Knowledge Base Bot** — Upload company docs, get instant Q&A
- **Domain Chatbot** — Upload Ramayana, legal texts, medical guides
- **Study Assistant** — Upload lecture notes, ask exam questions
- **CSV Data Explorer** — Upload structured data, query it in plain English

---

## Built For

**NYD 2026 Hackathon** — Designed to demonstrate rapid AI prototyping with flexible dataset integration.

---

## Contributors

- Shivam Shukla — [lucifer84670@gmail.com](mailto:lucifer84670@gmail.com)
- Mohit Sharma — [msharma42005@gmail.com](mailto:msharma42005@gmail.com)

---

## License

MIT License — free to use, modify, and distribute.
