import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import settings
from src.ingestion.loader import load_documents
from src.ingestion.chunker import chunk_documents
from src.retrieval.vector_store import VectorStore

docs = load_documents(settings.data_dir)
chunks = chunk_documents(docs, max_tokens=settings.max_chunk_tokens, overlap_tokens=settings.chunk_overlap_tokens)

store = VectorStore(dim=settings.embedding_dim)
store.build(chunks)

queries = [
    "How many sick days do I get?",
    "How much vacation time am I entitled to?",
    "How do I contact HR?",
]

for q in queries:
    print(f"\n=== Query: '{q}' ===")
    results = store.search(q, top_k=2)
    for chunk, score in results:
        print(f"  [{round(score,3)}] {chunk.section_title}: {chunk.text[:80]}...")