from typing import List, Tuple
from sentence_transformers import CrossEncoder

from src.ingestion.chunker import Chunk
from config import settings


class Reranker:
    _instance = None

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

    def rerank(self, query: str, candidates: List[Tuple[Chunk, float]], top_k: int = None, min_score: float = None):
        top_k = top_k or settings.final_top_k
        min_score = settings.rerank_min_score if min_score is None else min_score

        if not candidates:
            return []

        pairs = [(query, chunk.text) for chunk, _ in candidates]
        scores = self.model.predict(pairs)

        rescored = [(chunk, float(score)) for (chunk, _), score in zip(candidates, scores)]
        rescored.sort(key=lambda x: x[1], reverse=True)

        filtered = [item for item in rescored if item[1] >= min_score]

        if not filtered:
            filtered = rescored[:1]

        return filtered[:top_k]
