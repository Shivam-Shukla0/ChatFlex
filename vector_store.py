from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def createIndex(chunks):
    """Create TF-IDF vector index for text chunks."""
    vectorizer = TfidfVectorizer(stop_words='english')
    matrix = vectorizer.fit_transform(chunks)
    return {"vectorizer": vectorizer, "matrix": matrix, "chunks": chunks}


def searchContext(query, data, top_k=3):
    """Retrieve most relevant context chunks for a query using TF-IDF cosine similarity."""
    if data["index"] is None:
        return "No dataset loaded. Please upload a dataset first."

    vectorizer = data["index"]["vectorizer"]
    matrix = data["index"]["matrix"]
    chunks = data["index"]["chunks"]

    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, matrix).flatten()
    top_indices = np.argsort(scores)[::-1][:top_k]

    relevant = [chunks[i] for i in top_indices if scores[i] > 0]
    if not relevant:
        return "No relevant context found in the dataset for your question."

    return " ".join(relevant)
