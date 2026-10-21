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

questions = [
    "How many sick days do I get?",
    "How do I contact HR about this policy?",
    "What is the annual maximum coverage under the health insurance plan?",
    "How often must I change my password?",
]

for q in questions:
    print(f"\n=== Question: {q} ===")
    candidates = searcher.search(q, top_k=settings.hybrid_top_k)
    final = reranker.rerank(q, candidates, top_k=settings.final_top_k, min_score=settings.rerank_min_score)
    for i, (chunk, score) in enumerate(final, start=1):
        print(f"  [{i}] score={score:.3f} | doc={chunk.doc_title} | section={chunk.section_title}")