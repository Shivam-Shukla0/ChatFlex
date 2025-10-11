# rag_pipeline.py
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
from typing import List

MODEL_NAME = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')

class RAGIndex:
    def __init__(self, embed_model_name=MODEL_NAME):
        # SentenceTransformer handles device placement internally
        self.embed_model = SentenceTransformer(embed_model_name)
        self.index = None
        self.chunks = []
        self.dim = None

    def chunk_text(self, text: str, chunk_size: int = 300) -> List[str]:
        words = text.split()
        chunks = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
        return [c.strip() for c in chunks if c.strip()]

    def build(self, texts: List[str], chunk_size: int = 300):
        all_chunks = []
        for t in texts:
            all_chunks.extend(self.chunk_text(t, chunk_size))
        self.chunks = all_chunks
        # encode in one shot (small datasets)
        embeddings = self.embed_model.encode(self.chunks, show_progress_bar=False, convert_to_numpy=True)
        self.dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(self.dim)
        self.index.add(np.array(embeddings).astype('float32'))

    def build_from_fileobj(self, fileobj, chunk_size: int = 300, batch_size: int = 128, encoding: str = 'utf-8'):
        """Build index from a file-like object by streaming and encoding in batches.

        This avoids loading the entire file into memory and encodes chunks in batches for speed.
        - fileobj: file-like object opened in binary or text mode
        - chunk_size: number of words per chunk
        - batch_size: how many chunks to encode per model batch
        """
        def chunk_generator(fobj):
            # Read in binary-safe blocks and emit chunks of ~chunk_size words
            buffer_words = []
            read_bytes = 8192
            while True:
                data = fobj.read(read_bytes)
                if not data:
                    break
                if isinstance(data, bytes):
                    text_block = data.decode(encoding, errors='ignore')
                else:
                    text_block = data
                words = text_block.split()
                for w in words:
                    buffer_words.append(w)
                    if len(buffer_words) >= chunk_size:
                        yield ' '.join(buffer_words)
                        buffer_words = []
            if buffer_words:
                yield ' '.join(buffer_words)

        batch_chunks = []
        for chunk in chunk_generator(fileobj):
            batch_chunks.append(chunk)
            if len(batch_chunks) >= batch_size:
                embeddings = self.embed_model.encode(batch_chunks, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)
                if self.index is None:
                    self.dim = embeddings.shape[1]
                    self.index = faiss.IndexFlatL2(self.dim)
                self.index.add(np.array(embeddings).astype('float32'))
                self.chunks.extend(batch_chunks)
                batch_chunks = []

        # handle remaining
        if batch_chunks:
            embeddings = self.embed_model.encode(batch_chunks, batch_size=min(len(batch_chunks), batch_size), show_progress_bar=False, convert_to_numpy=True)
            if self.index is None:
                self.dim = embeddings.shape[1]
                self.index = faiss.IndexFlatL2(self.dim)
            self.index.add(np.array(embeddings).astype('float32'))
            self.chunks.extend(batch_chunks)

    def save(self, path='rag_index.pkl'):
        with open(path, 'wb') as f:
            pickle.dump({'chunks': self.chunks, 'dim': self.dim}, f)
        faiss.write_index(self.index, path + '.faiss')

    def load(self, path='rag_index.pkl'):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.chunks = data['chunks']
        self.dim = data['dim']
        self.index = faiss.read_index(path + '.faiss')

    def retrieve(self, query: str, top_k: int = 3):
        if self.index is None:
            raise ValueError('Index not built/loaded')
        q_emb = self.embed_model.encode([query], convert_to_numpy=True).astype('float32')
        D, I = self.index.search(q_emb, top_k)
        results = []
        for idx in I[0]:
            results.append(self.chunks[int(idx)])
        return results
