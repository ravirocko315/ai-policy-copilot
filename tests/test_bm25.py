import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import settings
from src.ingestion.loader import load_documents
from src.ingestion.chunker import chunk_documents
from src.retrieval.bm25_store import BM25Store

docs = load_documents(settings.data_dir)
chunks = chunk_documents(docs, max_tokens=settings.max_chunk_tokens, overlap_tokens=settings.chunk_overlap_tokens)

store = BM25Store()
store.build(chunks)

queries = [
    "medical certificate sick leave",
    "HR portal annual leave request",
    "bereavement",
]

for q in queries:
    print(f"\n=== Query: '{q}' ===")
    results = store.search(q, top_k=2)
    if not results:
        print("  (no keyword matches found)")
    for chunk, score in results:
        print(f"  [{round(score,3)}] {chunk.section_title}: {chunk.text[:80]}...")