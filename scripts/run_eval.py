import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import settings
from src.ingestion.loader import load_documents
from src.ingestion.chunker import chunk_documents
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_store import BM25Store
from src.retrieval.hybrid_search import HybridSearcher
from src.generation.rag_pipeline import PolicyCopilot
from src.evaluation.evaluator import load_eval_set, evaluate, print_report


def main():
    print("Loading and indexing documents...")
    docs = load_documents(settings.data_dir)
    chunks = chunk_documents(docs, max_tokens=settings.max_chunk_tokens, overlap_tokens=settings.chunk_overlap_tokens)

    vector_store = VectorStore(dim=settings.embedding_dim)
    vector_store.build(chunks)

    bm25_store = BM25Store()
    bm25_store.build(chunks)

    searcher = HybridSearcher(vector_store, bm25_store)
    copilot = PolicyCopilot(searcher)

    print("Loading evaluation set...")
    eval_set = load_eval_set("data/eval/eval_set.json")

    print(f"Running evaluation on {len(eval_set)} questions...\n")
    results = evaluate(copilot, eval_set)

    print_report(results)


if __name__ == "__main__":
    main()