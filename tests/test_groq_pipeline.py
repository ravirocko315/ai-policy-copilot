import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import settings
from src.ingestion.loader import load_documents
from src.ingestion.chunker import chunk_documents
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_store import BM25Store
from src.retrieval.hybrid_search import HybridSearcher
from src.generation.rag_pipeline import PolicyCopilot, GroqLLM

print("groq_api_key set:", bool(settings.groq_api_key))

docs = load_documents(settings.data_dir)
chunks = chunk_documents(
    docs,
    max_tokens=settings.max_chunk_tokens,
    overlap_tokens=settings.chunk_overlap_tokens,
)

vector_store = VectorStore(dim=settings.embedding_dim)
vector_store.build(chunks)

bm25_store = BM25Store()
bm25_store.build(chunks)

searcher = HybridSearcher(vector_store, bm25_store)
copilot = PolicyCopilot(searcher)

print("LLM being used:", type(copilot.llm).__name__)
assert isinstance(copilot.llm, GroqLLM), "Expected GroqLLM to be selected"

response = copilot.answer("How many sick days do I get?")
print("\nANSWER:\n", response.answer)
print("\nCITATIONS:")
for c in response.citations:
    print(" ", c.marker, c.doc_title, c.section_title)
