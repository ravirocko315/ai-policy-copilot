from typing import List, Tuple
from sentence_transformers import CrossEncoder

from src.ingestion.chunker import Chunk
from config import settings


class Reranker:
    _instance = None  # load the model only once, reuse it everywhere

    def __init__(self, model_name: str = None):
        model_name = model_name or settings.reranker_model
        print(f"Loading reranker model '{model_name}'... (first time takes a minute)")
        self.model = CrossEncoder(model_name)
        print("Reranker model loaded.")

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = Reranker()
        return cls._instance

    def rerank(self, query: str, candidates: List[Tuple[Chunk, float]], top_k: int = None):
        top_k = top_k or settings.final_top_k
        if not candidates:
            return []

        # Build (question, chunk_text) pairs -- the cross-encoder reads
        # both TOGETHER and outputs one relevance score per pair.
        pairs = [(query, chunk.text) for chunk, _ in candidates]
        scores = self.model.predict(pairs)

        # Attach the new score to each chunk and sort highest-first
        rescored = [(chunk, float(score)) for (chunk, _), score in zip(candidates, scores)]
        rescored.sort(key=lambda x: x[1], reverse=True)

        return rescored[:top_k]