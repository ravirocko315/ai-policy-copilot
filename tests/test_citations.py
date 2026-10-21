import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import settings
from src.ingestion.loader import load_documents
from src.ingestion.chunker import chunk_documents
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_store import BM25Store
from src.retrieval.hybrid_search import HybridSearcher
from src.retrieval.reranker import Reranker
from src.citations.citation_manager import build_citations, format_context_for_prompt

docs = load_documents(settings.data_dir)
chunks = chunk_documents(docs, max_tokens=settings.max_chunk_tokens, overlap_tokens=settings.chunk_overlap_tokens)

vector_store = VectorStore(dim=settings.embedding_dim)
vector_store.build(chunks)
bm25_store = BM25Store()
bm25_store.build(chunks)

searcher = HybridSearcher(vector_store, bm25_store)
reranker = Reranker.get()

question = "How many sick days do I get?"
candidates = searcher.search(question, top_k=5)
final_chunks = reranker.rerank(question, candidates, top_k=3)

print("=== CONTEXT BLOCK (what the AI will see) ===\n")
print(format_context_for_prompt(final_chunks))

print("\n\n=== CITATIONS (what the UI will show) ===\n")
citations = build_citations(final_chunks)
for c in citations:
    print(f"{c.marker} {c.doc_title} - {c.section_title} (score: {c.relevance_score})")
    print(f"    {c.snippet}")