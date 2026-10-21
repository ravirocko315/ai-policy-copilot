from collections import defaultdict
from typing import List, Tuple

from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_store import BM25Store
from src.ingestion.chunker import Chunk


def reciprocal_rank_fusion(dense_results, sparse_results, k: int = 60):
    """
    Combines two ranked lists of (chunk, score) tuples into one fused
    ranking, using only RANK POSITION (not raw scores) from each list.
    """
    fused_scores = defaultdict(float)
    chunk_lookup = {}

    for rank, (chunk, _) in enumerate(dense_results):
        fused_scores[chunk.chunk_id] += 1.0 / (k + rank + 1)
        chunk_lookup[chunk.chunk_id] = chunk

    for rank, (chunk, _) in enumerate(sparse_results):
        fused_scores[chunk.chunk_id] += 1.0 / (k + rank + 1)
        chunk_lookup[chunk.chunk_id] = chunk

    # Sort all chunks by their combined fused score, highest first
    ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return [(chunk_lookup[chunk_id], score) for chunk_id, score in ranked]


class HybridSearcher:
    def __init__(self, vector_store: VectorStore, bm25_store: BM25Store):
        self.vector_store = vector_store
        self.bm25_store = bm25_store

    def search(self, query: str, top_k: int = 5, fetch_k: int = 20) -> List[Tuple[Chunk, float]]:
        # Fetch a generous number of candidates from each system first
        dense_results = self.vector_store.search(query, top_k=fetch_k)
        sparse_results = self.bm25_store.search(query, top_k=fetch_k)

        # Merge them with RRF, then trim to the final top_k
        fused = reciprocal_rank_fusion(dense_results, sparse_results)
        return fused[:top_k]