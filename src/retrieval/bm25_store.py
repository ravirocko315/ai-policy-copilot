import re
from typing import List, Tuple
from rank_bm25 import BM25Okapi

from src.ingestion.chunker import Chunk

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> List[str]:
    """Turn text into a list of lowercase words/numbers, stripping punctuation."""
    return [t.lower() for t in _TOKEN_PATTERN.findall(text)]


class BM25Store:
    def __init__(self):
        self.bm25 = None
        self.chunks: List[Chunk] = []

    def build(self, chunks: List[Chunk]):
        self.chunks = chunks
        tokenized_corpus = [tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f"BM25Store built with {len(chunks)} chunks.")

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        query_tokens = tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        # Get indices sorted by score, highest first
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results = []
        for idx in ranked_indices[:top_k]:
            if scores[idx] > 0:  # skip chunks with zero keyword overlap
                results.append((self.chunks[idx], float(scores[idx])))
        return results