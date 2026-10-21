import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.embeddings.embedder import Embedder
import numpy as np

embedder = Embedder.get()

sentences = [
    "How many vacation days do I get?",
    "annual leave entitlement",
    "password policy requirements",
]

vectors = embedder.embed(sentences)

print("\nVector shape:", vectors.shape)  # expect (3, 384)

def cosine(a, b):
    return float(np.dot(a, b))  # already normalized, so dot product = cosine similarity

print("\nSimilarity (vacation vs annual leave):", round(cosine(vectors[0], vectors[1]), 3))
print("Similarity (vacation vs password policy):", round(cosine(vectors[0], vectors[2]), 3))