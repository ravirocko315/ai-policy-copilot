from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    _instance = None

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        print(f"Loading embedding model '{model_name}'... (first time takes a minute)")
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        print("Model loaded.")

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = Embedder()
        return cls._instance

    def embed(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        if is_query:
            texts = [f"Represent this sentence for searching relevant passages: {t}" for t in texts]

        vectors = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return np.asarray(vectors, dtype="float32")