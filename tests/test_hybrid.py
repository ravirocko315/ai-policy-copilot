import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import settings
from src.ingestion.loader import load_documents
from src.ingestion.chunker import chunk_documents
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_store import BM25Store
from src.retrieval.hybrid_search import HybridSearcher

docs = load_documents(settings.data_dir)
chunks = chunk_documents(docs, max_tokens=settings.max_chunk_tokens, overlap_tokens=settings.chunk_overlap_tokens)

vector_store = VectorStore(dim=settings.embedding_dim)
vector_store.build(chunks)

bm25_store = BM25Store()
bm25_store.build(chunks)

searcher = HybridSearcher(vector_store, bm25_store)

queries = [
    "How many sick days do I get?",
    "medical certificate",
    "time off work",  # paraphrase -- tests embeddings' strength
]

for q in queries:
    print(f"\n=== Query: '{q}' ===")
    results = searcher.search(q, top_k=3)
    for chunk, score in results:
        print(f"  [{round(score,4)}] {chunk.section_title}: {chunk.text[:70]}...")