import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import settings
from src.ingestion.loader import load_documents
from src.ingestion.chunker import chunk_documents
from src.embeddings.embedder import Embedder

docs = load_documents(settings.data_dir)
chunks = chunk_documents(docs, max_tokens=settings.max_chunk_tokens, overlap_tokens=settings.chunk_overlap_tokens)

print(f"Embedding {len(chunks)} chunks...")

embedder = Embedder.get()
texts = [c.text for c in chunks]
vectors = embedder.embed(texts, is_query=False)

print("Vectors shape:", vectors.shape)  # expect (5, 384) since we have 5 chunks

# Quick sanity check: which chunk is most similar to a sick-leave question?
import numpy as np
query_vec = embedder.embed(["How many sick days do I get?"], is_query=True)[0]

similarities = [float(np.dot(query_vec, v)) for v in vectors]
best_idx = int(np.argmax(similarities))

print("\nBest matching chunk for 'How many sick days do I get?':")
print("Section:", chunks[best_idx].section_title)
print("Similarity score:", round(similarities[best_idx], 3))
print("Text:", chunks[best_idx].text[:150])