import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import settings
from src.ingestion.loader import load_documents
from src.ingestion.chunker import chunk_documents
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_store import BM25Store
from src.retrieval.hybrid_search import HybridSearcher
from src.retrieval.reranker import Reranker

docs = load_documents(settings.data_dir)
chunks = chunk_documents(docs, max_tokens=settings.max_chunk_tokens, overlap_tokens=settings.chunk_overlap_tokens)

vector_store = VectorStore(dim=settings.embedding_dim)
vector_store.build(chunks)

bm25_store = BM25Store()
bm25_store.build(chunks)

searcher = HybridSearcher(vector_store, bm25_store)
reranker = Reranker.get()

queries = [
    "How many sick days do I get?",
    "time off work",
    "medical certificate",
]

for q in queries:
    print(f"\n=== Query: '{q}' ===")

    candidates = searcher.search(q, top_k=5)
    print("-- BEFORE reranking (hybrid order) --")
    for chunk, score in candidates[:3]:
        print(f"  [{round(score,4)}] {chunk.section_title}")

    final = reranker.rerank(q, candidates, top_k=3)
    print("-- AFTER reranking (cross-encoder order) --")
    for chunk, score in final:
        print(f"  [{round(score,3)}] {chunk.section_title}")