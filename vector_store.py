from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')


def createIndex(chunks):
    """Create FAISS vector index for text chunks."""
    embeddings = model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index


def searchContext(query, data, top_k=3):
    """Retrieve most relevant context chunks for a query."""
    if data["index"] is None:
        return "⚠️ No dataset loaded."
    query_emb = model.encode([query])
    distances, idxs = data["index"].search(np.array(query_emb), top_k)
    return " ".join([data["chunks"][i] for i in idxs[0]])
