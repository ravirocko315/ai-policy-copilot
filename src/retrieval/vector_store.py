from typing import List, Tuple
import faiss
import numpy as np

from src.ingestion.chunker import Chunk
from src.embeddings.embedder import Embedder


class VectorStore:
    def __init__(self, dim: int = 384):
        # IndexFlatIP = "Flat" (exact search, no approximation) using
        # "Inner Product" (dot product). Since our vectors are normalized,
        # dot product gives us cosine similarity -- exactly what we want.
        self.index = faiss.IndexFlatIP(dim)
        self.chunks: List[Chunk] = []

    def build(self, chunks: List[Chunk]):
        """Embed every chunk's text and store it in the FAISS index."""
        texts = [c.text for c in chunks]
        vectors = Embedder.get().embed(texts, is_query=False)
        self.index.add(vectors)
        self.chunks = chunks
        print(f"VectorStore built with {len(chunks)} chunks.")

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        """Embed the query and find the top_k most similar chunks."""
        query_vector = Embedder.get().embed([query], is_query=True)
        scores, indices = self.index.search(query_vector, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 if there aren't enough results
                continue
            results.append((self.chunks[idx], float(score)))
        return results